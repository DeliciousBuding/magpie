package redact

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// seedStandIns installs a temporary values/shapes table for the duration of t
// so Writer tests exercise shapeTail and Restore deterministically without
// leaking state into neighboring tests.
func seedStandIns(t *testing.T, vals, shps map[string]string) {
	t.Helper()
	mu.Lock()
	prevValues, prevShapes, prevKinds := values, shapes, kinds
	values = map[string]string{}
	shapes = map[string]string{}
	kinds = map[string]bool{}
	for k, v := range vals {
		values[k] = v
		if kind := kindOf(k); kind != "" {
			kinds[kind] = true
		}
	}
	for k, v := range shps {
		shapes[k] = v
		kinds[v] = true
	}
	mu.Unlock()
	t.Cleanup(func() {
		mu.Lock()
		values, shapes, kinds = prevValues, prevShapes, prevKinds
		mu.Unlock()
	})
}

func runSSEWriter(events ...string) string {
	rec := httptest.NewRecorder()
	rec.Header().Set("Content-Type", "text/event-stream")
	w := NewWriter(rec)
	for _, ev := range events {
		_, _ = w.Write([]byte(ev))
	}
	w.Finish()
	return rec.Body.String()
}

func geminiFunctionCallArgs(t *testing.T, sse string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(sse, "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		raw := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		var envelope struct {
			Response *struct {
				Candidates []struct {
					Content struct {
						Parts []struct {
							FunctionCall *struct {
								Name string         `json:"name"`
								Args map[string]any `json:"args"`
							} `json:"functionCall"`
							FunctionCallSnake *struct {
								Name string         `json:"name"`
								Args map[string]any `json:"args"`
							} `json:"function_call"`
						} `json:"parts"`
					} `json:"content"`
				} `json:"candidates"`
			} `json:"response"`
			Candidates []struct {
				Content struct {
					Parts []struct {
						FunctionCall *struct {
							Name string         `json:"name"`
							Args map[string]any `json:"args"`
						} `json:"functionCall"`
						FunctionCallSnake *struct {
							Name string         `json:"name"`
							Args map[string]any `json:"args"`
						} `json:"function_call"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal([]byte(raw), &envelope); err != nil {
			t.Fatalf("invalid JSON in SSE frame %q: %v", raw, err)
		}
		cands := envelope.Candidates
		if envelope.Response != nil && len(envelope.Response.Candidates) > 0 {
			cands = envelope.Response.Candidates
		}
		for _, c := range cands {
			for _, p := range c.Content.Parts {
				if p.FunctionCall != nil {
					out = append(out, p.FunctionCall.Args)
				}
				if p.FunctionCallSnake != nil {
					out = append(out, p.FunctionCallSnake.Args)
				}
			}
		}
	}
	return out
}

func TestWriterGeminiParallelFunctionCallsDoNotCarryTail(t *testing.T) {
	seedStandIns(t,
		map[string]string{
			"dummy_standin":       "real_d_secret",
			"yellow_standin":      "real_y_secret",
			"{{SECRET_abcd2345}}": "sk-real-secret-value",
		},
		map[string]string{
			"dummy_standin":  "SECRET",
			"yellow_standin": "SECRET",
		},
	)

	t.Run("three consecutive parallel calls keep trailing characters intact", func(t *testing.T) {
		got := runSSEWriter(
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":\"call_1\",\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/foo1.md\"}}}]},\"index\":0}]}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":\"call_2\",\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/foo2.md\"}}}]},\"index\":0}]}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":\"call_3\",\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/foo3.py\"}}}]},\"index\":0}]}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":\"call_4\",\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/ROADMAP.md\"}}}]},\"index\":0}]}\n\n",
		)
		calls := geminiFunctionCallArgs(t, got)
		want := []string{"/tmp/foo1.md", "/tmp/foo2.md", "/tmp/foo3.py", "/tmp/ROADMAP.md"}
		if len(calls) != len(want) {
			t.Fatalf("got %d calls, want %d:\n%s", len(calls), len(want), got)
		}
		for i, w := range want {
			if calls[i]["Path"] != w {
				t.Errorf("call[%d].Path = %q, want %q", i, calls[i]["Path"], w)
			}
		}
	})

	t.Run("nested objects and arrays inside functionCall.args do not shift tails", func(t *testing.T) {
		got := runSSEWriter(
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"batch_read\",\"args\":{\"meta\":{\"file\":\"/tmp/a.md\"},\"paths\":[\"/tmp/b.py\"]}}}]},\"index\":0}]}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"batch_read\",\"args\":{\"meta\":{\"file\":\"/tmp/c.md\"},\"paths\":[\"/tmp/d.py\"]}}}]},\"index\":0}]}\n\n",
		)
		calls := geminiFunctionCallArgs(t, got)
		if len(calls) != 2 {
			t.Fatalf("got %d calls, want 2:\n%s", len(calls), got)
		}
		if m0 := calls[0]["meta"].(map[string]any)["file"]; m0 != "/tmp/a.md" {
			t.Errorf("call[0].meta.file = %q, want /tmp/a.md", m0)
		}
		if m1 := calls[1]["meta"].(map[string]any)["file"]; m1 != "/tmp/c.md" {
			t.Errorf("call[1].meta.file = %q, want /tmp/c.md", m1)
		}
		if p0 := calls[0]["paths"].([]any)[0]; p0 != "/tmp/b.py" {
			t.Errorf("call[0].paths[0] = %q, want /tmp/b.py", p0)
		}
		if p1 := calls[1]["paths"].([]any)[0]; p1 != "/tmp/d.py" {
			t.Errorf("call[1].paths[0] = %q, want /tmp/d.py", p1)
		}
	})

	t.Run("code assist wrapped response and snake_case function_call do not shift tails", func(t *testing.T) {
		got := runSSEWriter(
			"data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"function_call\":{\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/ca1.md\"}}}]},\"index\":0}]}}\n\n",
			"data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"function_call\":{\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/ca2.md\"}}}]},\"index\":0}]}}\n\n",
		)
		calls := geminiFunctionCallArgs(t, got)
		if len(calls) != 2 || calls[0]["Path"] != "/tmp/ca1.md" || calls[1]["Path"] != "/tmp/ca2.md" {
			t.Fatalf("code assist / snake_case calls corrupted: %#v", calls)
		}
	})

	t.Run("multiple functionCalls in one frame restore placeholders in place", func(t *testing.T) {
		got := runSSEWriter(
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"set_env\",\"args\":{\"token\":\"{{SECRET_abcd2345}}\",\"path\":\"/tmp/one.md\"}}},{\"functionCall\":{\"name\":\"set_env\",\"args\":{\"token\":\"dummy_standin\",\"path\":\"/tmp/two.md\"}}}]},\"index\":0}]}\n\n",
		)
		calls := geminiFunctionCallArgs(t, got)
		if len(calls) != 2 {
			t.Fatalf("got %d calls, want 2:\n%s", len(calls), got)
		}
		if calls[0]["token"] != "sk-real-secret-value" || calls[0]["path"] != "/tmp/one.md" {
			t.Errorf("call[0] = %#v", calls[0])
		}
		if calls[1]["token"] != "real_d_secret" || calls[1]["path"] != "/tmp/two.md" {
			t.Errorf("call[1] = %#v", calls[1])
		}
	})

	t.Run("different tool names sharing a parameter name and keepalive comments between calls", func(t *testing.T) {
		got := runSSEWriter(
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":\"call_a\",\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/first.md\"}}}]},\"index\":0}]}\n\n",
			": keepalive\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"id\":\"call_b\",\"name\":\"stat_file\",\"args\":{\"Path\":\"/tmp/second.md\"}}}]},\"index\":0}]}\n\n",
		)
		calls := geminiFunctionCallArgs(t, got)
		if len(calls) != 2 || calls[0]["Path"] != "/tmp/first.md" || calls[1]["Path"] != "/tmp/second.md" {
			t.Fatalf("different-tool parallel calls corrupted: %#v\n%s", calls, got)
		}
		if !strings.Contains(got, `"id":"call_a","name":"read_file"`) || !strings.Contains(got, `"id":"call_b","name":"stat_file"`) {
			t.Errorf("functionCall id/name unexpectedly modified:\n%s", got)
		}
	})

	t.Run("functionResponse.response atomic fields do not shift tails across frames", func(t *testing.T) {
		got := runSSEWriter(
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionResponse\":{\"name\":\"lookup\",\"response\":{\"output\":\"/tmp/res1.md\"}}}]},\"index\":0}]}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionResponse\":{\"name\":\"lookup\",\"response\":{\"output\":\"/tmp/res2.md\"}}}]},\"index\":0}]}\n\n",
		)
		if !strings.Contains(got, `"/tmp/res1.md"`) || !strings.Contains(got, `"/tmp/res2.md"`) {
			t.Fatalf("functionResponse.response corrupted:\n%s", got)
		}
	})

	t.Run("streamed Gemini text deltas still join split placeholders across frames", func(t *testing.T) {
		got := runSSEWriter(
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"key is {{SECRET_\"}]},\"index\":0}]}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"abcd2345}} and dum\"}]},\"index\":0}]}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"my_standin done\"}]},\"index\":0}]}\n\n",
		)
		if !strings.Contains(got, "sk-real-secret-value") {
			t.Errorf("split brace placeholder in Gemini text delta was not restored:\n%s", got)
		}
		if !strings.Contains(got, "real_d_secret") {
			t.Errorf("split stand-in in Gemini text delta was not restored:\n%s", got)
		}
	})

	t.Run("streamed OpenAI and Anthropic tool argument deltas still join split placeholders", func(t *testing.T) {
		gotAnthropic := runSSEWriter(
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"k\\\":\\\"{{SECRET_\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"abcd2345}}}\\\"\"}}\n\n",
		)
		if !strings.Contains(gotAnthropic, "sk-real-secret-value") {
			t.Errorf("Anthropic split partial_json placeholder not restored:\n%s", gotAnthropic)
		}

		gotOpenAI := runSSEWriter(
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"k\\\":\\\"dum\"}}]}}]}\n\n",
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"my_standin\\\"}\"}}]}}]}\n\n",
		)
		if !strings.Contains(gotOpenAI, "real_d_secret") {
			t.Errorf("OpenAI split tool_calls arguments stand-in not restored:\n%s", gotOpenAI)
		}
	})
}
