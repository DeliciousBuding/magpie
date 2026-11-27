package catalog

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func TestFetchGeminiPageLimit(t *testing.T) {
	for _, pages := range []int{20, 21} {
		t.Run(strconv.Itoa(pages), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				offset, _ := strconv.Atoi(r.URL.Query().Get("pageToken"))
				page := offset + 1
				next := ""
				if page < pages {
					next = strconv.Itoa(page)
				}
				fmt.Fprintf(w, `{"models":[{"name":"models/m-%d","supportedGenerationMethods":["generateContent"]}],"nextPageToken":%q}`, page, next)
			}))
			defer srv.Close()

			ms, at, err := FetchGemini(t.Context(), srv.URL, "", nil)
			if at != srv.URL+"/models" {
				t.Fatalf("list URL = %q", at)
			}
			if pages == 20 {
				if err != nil || len(ms) != 20 || ms[19].ID != "m-20" {
					t.Fatalf("complete list: got %d models, err %v", len(ms), err)
				}
				return
			}
			if len(ms) != 0 || err == nil || !strings.Contains(err.Error(), "still paging after 20 pages") {
				t.Fatalf("unfinished list: got %d models, err %v", len(ms), err)
			}
		})
	}
}

func TestFetchGeminiRepeatedToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"models":[{"name":"models/m","supportedGenerationMethods":["generateContent"]}],"nextPageToken":"same"}`)
	}))
	defer srv.Close()
	ms, _, err := FetchGemini(t.Context(), srv.URL, "", nil)
	if len(ms) != 0 || err == nil || !strings.Contains(err.Error(), "repeated page token") {
		t.Fatalf("stalled list: got %d models, err %v", len(ms), err)
	}
}

func TestFetchURLRejectsUnfollowedNextPageToken(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{"nextPageToken alone", `{"models":[{"name":"models/m-1","supportedGenerationMethods":["generateContent"]}],"nextPageToken":"page-2"}`},
		{"nextPageToken with has_more", `{"models":[{"name":"models/m-1","supportedGenerationMethods":["generateContent"]}],"nextPageToken":"page-2","has_more":true,"last_id":"models/m-1"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Query().Get("after_id") != "" {
					fmt.Fprint(w, `{"models":[],"has_more":false}`)
					return
				}
				fmt.Fprint(w, tc.body)
			}))
			defer srv.Close()
			ms, err := FetchURL(t.Context(), srv.URL+"/models", "", false, nil)
			if len(ms) != 0 || err == nil || !strings.Contains(err.Error(), "nextPageToken") {
				t.Fatalf("truncated ModelsURL list: got %d models, err %v", len(ms), err)
			}
		})
	}
}
