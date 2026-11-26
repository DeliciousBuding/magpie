package redact

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// withStandIn makes the stand-in the issue's test made (#7837): a made-up
// key beginning with an ordinary letter, so a text ending in d may be the
// start of one, and the Writer holds that d for the next event.
func withStandIn(t *testing.T) {
	t.Helper()
	mu.Lock()
	prevValues, prevShapes := values, shapes
	values = map[string]string{"dummy_standin": "real_secret"}
	shapes = map[string]string{"dummy_standin": "SECRET"}
	mu.Unlock()
	t.Cleanup(func() {
		mu.Lock()
		values, shapes = prevValues, prevShapes
		mu.Unlock()
	})
}

func throughWriter(evs ...string) string {
	rec := httptest.NewRecorder()
	rec.Header().Set("Content-Type", "text/event-stream")
	w := NewWriter(rec)
	w.WriteHeader(200)
	for _, e := range evs {
		w.Write([]byte(e))
	}
	w.Finish()
	return rec.Body.String()
}

// Gemini's :streamGenerateContent?alt=sse sends each of a turn's parallel
// functionCalls whole, in an event of its own, at the same path
// (candidates.0.content.parts.0.functionCall.args.Path). The d at the end
// of one call's path was held for a placeholder's start and moved to the
// start of the next call's: /tmp/foo1.m, d/tmp/foo2.m (DeliciousBuding,
// #7837; the issue's bytes as Gemini sent them). Each call goes out as it
// came, and a placeholder in each is put back in that call.
func TestWriterGeminiParallelCallsKeepTheirArgs(t *testing.T) {
	withStandIn(t)
	healthy := "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/foo1.md\"}}}]},\"finishReason\":null,\"index\":0}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/foo2.md\"}}}]},\"finishReason\":null,\"index\":0}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"read_file\",\"args\":{\"Path\":\"/tmp/foo3.py\"}}}]},\"finishReason\":null,\"index\":0}]}\n\n"
	evs := strings.SplitAfter(healthy, "\n\n")
	if got := throughWriter(evs...); got != healthy {
		t.Fatalf("parallel functionCalls changed:\n got %s\nwant %s", got, healthy)
	}
	// Code Assist's wrapper, snake_case, nested args, a functionResponse
	wrapped := []string{
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"function_call":{"name":"edit","args":{"meta":{"file":"/tmp/ca1.md"},"paths":["/tmp/a.md","/tmp/b.md"]}}}]},"index":0}]}}` + "\n\n",
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"function_call":{"name":"edit","args":{"meta":{"file":"/tmp/ca2.md"},"paths":["/tmp/c.md","/tmp/d.md"]}}}]},"index":0}]}}` + "\n\n",
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionResponse":{"name":"edit","response":{"out":"wrote /tmp/x.md"}}}]},"index":0}]}` + "\n\n",
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionResponse":{"name":"edit","response":{"out":"wrote /tmp/y.md"}}}]},"index":0}]}` + "\n\n",
	}
	if got, want := throughWriter(wrapped...), strings.Join(wrapped, ""); got != want {
		t.Fatalf("Code Assist calls changed:\n got %s\nwant %s", got, want)
	}
	// a placeholder in each call is put back in that call
	const v1, v2 = "Hunter2Alpha-77", "Hunter2Bravo-88"
	p1, p2 := placeholder("SECRET", v1), placeholder("SECRET", v2)
	call := func(path string) string {
		return `data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"read_file","args":{"Path":"` + path + `"}}}]},"index":0}]}` + "\n\n"
	}
	got := throughWriter(call("/tmp/"+p1+"/a.md"), call("/tmp/"+p2+"/b.md"))
	if want := call("/tmp/"+v1+"/a.md") + call("/tmp/"+v2+"/b.md"); got != want {
		t.Fatalf("placeholders in calls:\n got %s\nwant %s", got, want)
	}
}

// withIPStandIn registers a real public-ip- stand-in (foldStandIn("IP") == true)
// so uppercase "STOP" ending in "P" triggers partialTail, reproducing the
// Gemini finishReason "STO"/"PSTOP" split.
func withIPStandIn(t *testing.T) {
	t.Helper()
	mu.Lock()
	prevValues, prevShapes := values, shapes
	values = map[string]string{"public-ip-kvxqmrtz": "11.71.52.83"}
	shapes = map[string]string{"public-ip-kvxqmrtz": "IP"}
	mu.Unlock()
	t.Cleanup(func() {
		mu.Lock()
		values, shapes = prevValues, prevShapes
		mu.Unlock()
	})
}

func TestWriterGeminiReplyMetadata(t *testing.T) {
	withIPStandIn(t)
	const (
		candidate = `{"candidates":[{"index":0,"finishReason":"STOP","finishMessage":"hit public-ip-kvxqmrtz ends in P","safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"LOW"}]}],"usageMetadata":{"totalTokenCount":10}}`
		partial   = `{"candidates":[{"index":0,"finishReason":"MAX_TOKENS","continuationToken":"token ends in P"}],"usageMetadata":{"totalTokenCount":10,"trafficType":"ON_DEMAND"},"modelVersion":"gemini-test-P","responseId":"resp-P"}`
		blocked   = `{"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"see public-ip-kvxqmrtz ends in P","safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"LOW"}]}}`
	)
	candidateRestored := strings.Replace(candidate, "public-ip-kvxqmrtz", "11.71.52.83", 1)
	blockedRestored := strings.Replace(blocked, "public-ip-kvxqmrtz", "11.71.52.83", 1)
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"native candidate", candidate, candidateRestored},
		{"Code Assist candidate", `{"response":` + candidate + `}`, `{"response":` + candidateRestored + `}`},
		{"native metadata ending in P", partial, partial},
		{"Code Assist metadata ending in P", `{"response":` + partial + `,"traceId":"trace-P"}`, `{"response":` + partial + `,"traceId":"trace-P"}`},
		{"native prompt feedback", blocked, blockedRestored},
		{"Code Assist prompt feedback", `{"response":` + blocked + `}`, `{"response":` + blockedRestored + `}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ev := "data: " + tc.body + "\n\n"
			want := "data: " + tc.want + "\n\n"
			if got := throughWriter(ev, ev); got != want+want {
				t.Errorf("stream reply changed:\n got %s\nwant %s", got, want+want)
			}
		})
	}
	// Google ADK streams content.parts[] without a candidates[] wrapper; full
	// functionCall.args must still stay in their own event.
	adk1 := `data: {"content":{"role":"model","parts":[{"functionCall":{"name":"read","args":{"file":"first.P"}}}]}}` + "\n\n"
	adk2 := `data: {"content":{"role":"model","parts":[{"functionCall":{"name":"read","args":{"file":"next.go"}}}]}}` + "\n\n"
	if got := throughWriter(adk1, adk2); got != adk1+adk2 {
		t.Errorf("ADK functionCall args changed:\n got %s\nwant %s", got, adk1+adk2)
	}
	// Two Gemini partialArgs with >128-byte jsonPaths reordered between frames
	// must keep their full jsonPath identity without falling back to array index.
	longA := "$." + strings.Repeat("a", 130)
	longB := "$." + strings.Repeat("b", 130)
	p1 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"edit","partialArgs":[{"jsonPath":"` + longA + `","stringValue":"public-","willContinue":true},{"jsonPath":"` + longB + `","stringValue":"left","willContinue":true}]}}]}}]}` + "\n\n"
	p2 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"edit","partialArgs":[{"jsonPath":"` + longB + `","stringValue":"right","willContinue":false},{"jsonPath":"` + longA + `","stringValue":"ip-kvxqmrtz","willContinue":false}]}}]}}]}` + "\n\n"
	gotLong := throughWriter(p1, p2)
	if !strings.Contains(gotLong, `"stringValue":"11.71.52.83"`) || !strings.Contains(gotLong, `"stringValue":"right"`) {
		t.Errorf("long jsonPath reordered partialArgs corrupted: %s", gotLong)
	}
	// Duplicate jsonPath in the next event must stay ambiguous rather than
	// preferring array slot 1 over slot 0 (which would reorder STOP+RED+BLUE
	// into STOREDPBLUE).
	dup1 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"edit","partialArgs":[{"jsonPath":"$.other","stringValue":"ok","willContinue":true},{"jsonPath":"$.text","stringValue":"STOP","willContinue":true}]}}]}}]}` + "\n\n"
	dup2 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"edit","partialArgs":[{"jsonPath":"$.text","stringValue":"RED","willContinue":true},{"jsonPath":"$.text","stringValue":"BLUE","willContinue":false}]}}]}}]}` + "\n\n"
	if got := throughWriter(dup1, dup2); got != dup1+dup2 {
		t.Errorf("duplicate partialArgs jsonPath reordered characters:\n got %s\nwant %s", got, dup1+dup2)
	}
	// A dotted sibling key ("content.parts": null) on the candidate must not
	// turn real content.parts functionCall args into a streamed field.
	sib1 := `data: {"candidates":[{"index":0,"content.parts":null,"content":{"parts":[{"functionCall":{"name":"read_file","args":{"file":"first.P"}}}]}}]}` + "\n\n"
	sib2 := `data: {"candidates":[{"index":0,"content.parts":null,"content":{"parts":[{"functionCall":{"name":"read_file","args":{"file":"next.go"}}}]}}]}` + "\n\n"
	if got := throughWriter(sib1, sib2); got != sib1+sib2 {
		t.Errorf("dotted sibling key content.parts broke Gemini functionCall args:\n got %s\nwant %s", got, sib1+sib2)
	}
	// Duplicate willContinue keys follow JSON last-key-wins: true then false
	// finishes the partialArg so STOP does not carry P into NEXT.
	wc1 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"edit","partialArgs":[{"jsonPath":"$.x","stringValue":"STOP","willContinue":true,"willContinue":false}]}}]}}]}` + "\n\n"
	wc2 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"edit","partialArgs":[{"jsonPath":"$.x","stringValue":"NEXT","willContinue":false}]}}]}}]}` + "\n\n"
	if got := throughWriter(wc1, wc2); got != wc1+wc2 {
		t.Errorf("duplicate willContinue true->false carried tail:\n got %s\nwant %s", got, wc1+wc2)
	}
	// Chat tool_calls by id without index: call-a followed by omitted-id chunk
	// ending in STOP, then a brand-new call-b with "{}", must keep call-b's
	// JSON untouched rather than carrying call-a's P onto call-b.
	idA1 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-a","function":{"arguments":"{\"file\":\""}}]}}]}` + "\n\n"
	idA2 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"STOP"}}]}}]} ` + "\n\n"
	idB := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-b","function":{"arguments":"{}"}}]}}]}` + "\n\n"
	idA3 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-a","function":{"arguments":".go\"}"}}]}}]}` + "\n\n"
	if got := throughWriter(idA1, idA2, idB, idA3); got != idA1+idA2+idB+idA3 {
		t.Errorf("omitted-id call-a leaked tail onto call-b:\n got %s\nwant %s", got, idA1+idA2+idB+idA3)
	}
	// call-a (id only) -> call-b (index:0 + id) -> omitted id STOP -> call-a -> call-b
	// must never carry STOP's P onto call-a.
	mix1 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-a","function":{"arguments":"{\"n\":"}}]}}]}` + "\n\n"
	mix2 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-b","function":{"arguments":"{\"file\":\""}}]}}]}` + "\n\n"
	mix3 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"STOP"}}]}}]} ` + "\n\n"
	mix4 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-a","function":{"arguments":"1}"}}]}}]}` + "\n\n"
	mix5 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-b","function":{"arguments":".go\"}"}}]}}]}` + "\n\n"
	if got := throughWriter(mix1, mix2, mix3, mix4, mix5); got != mix1+mix2+mix3+mix4+mix5 {
		t.Errorf("mixed index/id calls leaked tail:\n got %s\nwant %s", got, mix1+mix2+mix3+mix4+mix5)
	}
	// Reordered choices where two tool_calls share index:0 but declare distinct
	// explicit ids (call-a vs call-b) must never carry call-a's P onto call-b.
	re1 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-a","function":{"arguments":"{\"file\":\"STOP"}}]}}]} ` + "\n\n"
	re2 := `data: {"choices":[{"index":1,"delta":{"content":"x"}},{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-b","function":{"arguments":"{}"}}]}}]}` + "\n\n"
	re3 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-a","function":{"arguments":".go\"}"}}]}}]}` + "\n\n"
	if got := throughWriter(re1, re2, re3); got != re1+re2+re3 {
		t.Errorf("distinct tool_call ids sharing index:0 leaked tail across reordered choices:\n got %s\nwant %s", got, re1+re2+re3)
	}
	// An omitted-id continuation on index:0 followed by a new call declaring a
	// fresh id on index:0 must not carry the tail into the new call.
	om1 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-a","function":{"arguments":"{\"file\":\""}}]}}]}` + "\n\n"
	om2 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"STOP"}}]}}]} ` + "\n\n"
	om3 := `data: {"choices":[{"index":1,"delta":{"content":"x"}},{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-b","function":{"arguments":"{}"}}]}}]}` + "\n\n"
	om4 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-a","function":{"arguments":".go\"}"}}]}}]}` + "\n\n"
	if got := throughWriter(om1, om2, om3, om4); got != om1+om2+om3+om4 {
		t.Errorf("omitted-id chunk on index:0 leaked tail into new call-b:\n got %s\nwant %s", got, om1+om2+om3+om4)
	}
	// Two Gemini functionCalls with distinct explicit ids sharing partialArgs
	// jsonPath across reordered candidates must stay isolated.
	fc1 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"id":"fc_a","name":"edit","partialArgs":[{"jsonPath":"$.file","stringValue":"STOP","willContinue":true}]}}]}}]}` + "\n\n"
	fc2 := `data: {"candidates":[{"index":1,"content":{"parts":[{"text":"x"}]}},{"index":0,"content":{"parts":[{"functionCall":{"id":"fc_b","name":"edit","partialArgs":[{"jsonPath":"$.file","stringValue":"README","willContinue":false}]}}]}}]}` + "\n\n"
	fc3 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"id":"fc_a","name":"edit","partialArgs":[{"jsonPath":"$.file","stringValue":".go","willContinue":false}]}}]}}]}` + "\n\n"
	if got := throughWriter(fc1, fc2, fc3); got != fc1+fc2+fc3 {
		t.Errorf("distinct Gemini functionCall ids sharing partialArgs jsonPath leaked tail:\n got %s\nwant %s", got, fc1+fc2+fc3)
	}
}

func TestWriterProtocolsKeepCompleteFields(t *testing.T) {
	withStandIn(t)
	const (
		gemini          = `{"candidates":[{"index":0,"groundingMetadata":{"webSearchQueries":["word"]}}],"usageMetadata":{"trafficType":"ON_DEMAND"},"responseId":"reply-d"}`
		assist          = `{"response":{"candidates":[{"index":0,"finish_reason":"STOP"}],"usage_metadata":{"traffic_type":"ON_DEMAND"},"model_version":"model-d","response_id":"reply-d"}}`
		chat            = `{"object":"chat.completion.chunk","system_fingerprint":"fp_d","choices":[{"index":0,"delta":{"content":"ok","annotations":[{"type":"url_citation","url_citation":{"title":"doc d"}}],"reasoning_details":[{"index":0,"format":"anthropic-claude-v1-d"}]},"logprobs":{"content":[{"token":"word"}]}}]}`
		chatNoObj       = `{"id":"chatcmpl-1","system_fingerprint":"fp_d","choices":[{"index":0,"delta":null,"finish_reason":"stop"}]}`
		chatUsageOnly   = `{"id":"chatcmpl-1","system_fingerprint":"fp_d","choices":[],"usage":{"total_tokens":2}}`
		msgLifecycle    = `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"word"}}`
		messages        = `{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","title":"document d","cited_text":"word"}}}`
		blocks          = `{"type":"content_block_delta","index" : N,"delta":{"type":"text_delta","text":"word"}}`
		responseContent = `{"type":"response.output_text.delta","item_id":"item","output_index":0,"content_index" : N,"delta":"word"}`
		responses       = `{"type":"response.output_text.annotation.added","item_id":"item","output_index":0,"content_index":0,"annotation_index":N,"annotation":{"title":"document d"}}`
		summary         = `{"type":"response.reasoning_summary_text.delta","item_id":"item","output_index":0,"summary_index" : N,"delta":"word"}`
		shellCmd        = `{"type":"response.shell_call_output_content.delta","item_id":"sh_1","output_index":0,"command_index":N,"delta":{"stdout":"word"}}`
		audioDelta      = `{"type":"response.audio.delta","delta":"word"}`
		outputAudio     = `{"type":"response.output_audio.delta","delta":"word"}`
		partialImg      = `{"type":"response.image_generation_call.partial_image","partial_image_b64":"word"}`
	)
	for _, tc := range []struct{ name, first, second string }{
		{"Gemini grounding and metadata", gemini, gemini},
		{"Code Assist snake case", assist, assist},
		{"Chat metadata, annotations and logprobs", chat, chat},
		{"Chat without object and null delta", chatNoObj, chatNoObj},
		{"Chat empty choices usage chunk", chatUsageOnly, chatUsageOnly},
		{"Messages lifecycle block start", msgLifecycle, msgLifecycle},
		{"Messages citations", messages, messages},
		{"Messages distinct content blocks", strings.Replace(blocks, "N", "0", 1), strings.Replace(blocks, "N", "1", 1)},
		{"Responses distinct content parts", strings.Replace(responseContent, "N", "0", 1), strings.Replace(responseContent, "N", "1", 1)},
		{"Responses annotations", strings.Replace(responses, "N", "0", 1), strings.Replace(responses, "N", "1", 1)},
		{"Responses summary parts", strings.Replace(summary, "N", "0", 1), strings.Replace(summary, "N", "1", 1)},
		{"Responses distinct shell commands", strings.Replace(shellCmd, "N", "0", 1), strings.Replace(shellCmd, "N", "1", 1)},
		{"Responses audio delta", audioDelta, audioDelta},
		{"Responses output_audio delta", outputAudio, outputAudio},
		{"Responses partial image", partialImg, partialImg},
	} {
		t.Run(tc.name, func(t *testing.T) {
			first, second := "data: "+tc.first+"\r\n\r\n", "data: "+tc.second+"\r\n\r\n"
			want := first + second
			pieces := make([]string, 0, len(want)/7+1)
			for rest := want; len(rest) > 0; {
				n := min(7, len(rest))
				pieces = append(pieces, rest[:n])
				rest = rest[n:]
			}
			if got := throughWriter(pieces...); got != want {
				t.Errorf("complete fields moved between events:\n got %s\nwant %s", got, want)
			}
		})
	}
	// Distinct SSE event names with the same JSON "type" must not share a tail,
	// even when one is "response.output_text.delta" and the other is explicit
	// "event: message".
	evAudit := "event: audit\ndata: " + `{"type":"audit","text":"word"}` + "\n\n"
	evMsg := "event: message\ndata: " + `{"type":"audit","text":"next"}` + "\n\n"
	if got := throughWriter(evAudit, evMsg); got != evAudit+evMsg {
		t.Errorf("tail crossed between distinct SSE event names:\n got %s\nwant %s", got, evAudit+evMsg)
	}
	evResp := "event: response.output_text.delta\ndata: " + `{"type":"response.output_text.delta","delta":"word"}` + "\n\n"
	evRespEmpty := `data: {"type":"response.output_text.delta","delta":""}` + "\n\n"
	evRespMsg := "event: message\ndata: " + `{"type":"response.output_text.delta","delta":"next"}` + "\n\n"
	if got := throughWriter(evResp, evRespMsg); got != evResp+evRespMsg {
		t.Errorf("tail crossed between explicit response event and event: message:\n got %s\nwant %s", got, evResp+evRespMsg)
	}
	if got := throughWriter(evResp, evRespEmpty, evRespMsg); got != evResp+evRespEmpty+evRespMsg {
		t.Errorf("omitted event: middle frame bridged explicit response event to event: message:\n got %s\nwant %s", got, evResp+evRespEmpty+evRespMsg)
	}
	// Trailing whitespace in SSE event: is significant per WHATWG SSE, so
	// "event: message " is distinct from omitted event: (default "message").
	evTrail := "event: message \ndata: " + `{"type":"x","text":"word"}` + "\n\n"
	evDef := `data: {"type":"x","text":"next"}` + "\n\n"
	if got := throughWriter(evTrail, evDef); got != evTrail+evDef {
		t.Errorf("trailing space in event: message collapsed to default event:\n got %s\nwant %s", got, evTrail+evDef)
	}
	// Non-array delta object with distinct index must not share a tail.
	d0 := `data: {"delta":{"index":0,"text":"word"}}` + "\n\n"
	d1 := `data: {"delta":{"index":1,"text":"next"}}` + "\n\n"
	if got := throughWriter(d0, d1); got != d0+d1 {
		t.Errorf("tail crossed between indexed delta objects:\n got %s\nwant %s", got, d0+d1)
	}
	// Truncated JSON in frame 2 must flush the held frame 1 whole and pass
	// frame 2 verbatim without rewriting SSE id: headers.
	good1 := `data: {"choices":[{"index":0,"delta":{"content":"hello dummy_"}}]}` + "\n\n"
	trunc2 := "id: dummy_standin\ndata: " + `{"choices":[{"index":0,"delta":{"content":"standin"}}` + "\n\n"
	if got := throughWriter(good1, trunc2); got != good1+trunc2 {
		t.Errorf("truncated JSON frame dropped tail or rewrote header:\n got %s\nwant %s", got, good1+trunc2)
	}
	// Valid JSON keys containing control bytes (\u001f, \u0000#7) must never
	// collide with nested object paths or array identity markers.
	ctl1 := `data: {"a":{"b":"word"}}` + "\n\n"
	ctl2 := `data: {"a\u001fb":"next"}` + "\n\n"
	if got := throughWriter(ctl1, ctl2); got != ctl1+ctl2 {
		t.Errorf("control-byte JSON key collided with nested path:\n got %s\nwant %s", got, ctl1+ctl2)
	}
	arr1 := `data: {"items":[{"index":7,"text":"word"}]}` + "\n\n"
	arr2 := `data: {"items":{"\u0000#7":{"text":"next"}}}` + "\n\n"
	if got := throughWriter(arr1, arr2); got != arr1+arr2 {
		t.Errorf("literal key collided with array index marker:\n got %s\nwant %s", got, arr1+arr2)
	}
	// Literal dotted key "delta.content" must not collide with nested delta.content.
	dot1 := `data: {"choices":[{"index":0,"delta":{"content":"dummy_"}}]}` + "\n\n"
	dot2 := `data: {"choices":[{"index":0,"delta":{"content":"standin"},"delta.content":"meta"}]}` + "\n\n"
	wantDot := `data: {"choices":[{"index":0,"delta":{"content":""}}]}` + "\n\n" +
		`data: {"choices":[{"index":0,"delta":{"content":"real_secret"},"delta.content":"meta"}]}` + "\n\n"
	if got := throughWriter(dot1, dot2); got != wantDot {
		t.Errorf("literal dotted sibling collided with delta.content:\n got %s\nwant %s", got, wantDot)
	}
	dotOnly2 := `data: {"choices":[{"index":0,"delta.content":"meta"}]}` + "\n\n"
	if got := throughWriter(dot1, dotOnly2); got != dot1+dotOnly2 {
		t.Errorf("literal dotted key absorbed held tail:\n got %s\nwant %s", got, dot1+dotOnly2)
	}
	// Nested "parts" inside Gemini functionCall.args must stay complete.
	nested1 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"write","args":{"parts":[{"text":"word"}]}}}]}}]}` + "\n\n"
	nested2 := `data: {"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"write","args":{"parts":[{"text":"next"}]}}}]}}]}` + "\n\n"
	if got := throughWriter(nested1, nested2); got != nested1+nested2 {
		t.Errorf("nested parts inside functionCall.args changed:\n got %s\nwant %s", got, nested1+nested2)
	}
}

func TestWriterRestoresProtocolAndNativeDeltas(t *testing.T) {
	withStandIn(t)
	for _, tc := range []struct {
		name string
		SSE  string
		body string
		path []any
	}{
		{"chat reasoning_content", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"PIECE"}}]}`, []any{"choices", 0, "delta", "reasoning_content"}},
		{"chat reasoning details summary", "", `{"choices":[{"index":0,"delta":{"reasoning_details":[{"index":0,"summary":"PIECE"}]}}]}`, []any{"choices", 0, "delta", "reasoning_details", 0, "summary"}},
		{"chat mistral content thinking string", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":"PIECE"}]}}]}`, []any{"choices", 0, "delta", "content", 0, "thinking"}},
		{"chat mistral content thinking parts", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"PIECE"}]}]}}]}`, []any{"choices", 0, "delta", "content", 0, "thinking", 0, "text"}},
		{"chat litellm thinking_blocks", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"thinking_blocks":[{"index":0,"type":"thinking","thinking":"PIECE"}]}}]}`, []any{"choices", 0, "delta", "thinking_blocks", 0, "thinking"}},
		{"chat litellm provider_specific_fields", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"provider_specific_fields":{"compaction_delta":{"content":"PIECE"}}}}]}`, []any{"choices", 0, "delta", "provider_specific_fields", "compaction_delta", "content"}},
		{"chat extra_content custom extension", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"extra_content":{"x_text":"PIECE"}}}]}`, []any{"choices", 0, "delta", "extra_content", "x_text"}},
		{"chat literal dotted message.text", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"message.text":"PIECE"}]}`, []any{"choices", 0, "message.text"}},
		{"chat literal dotted parent message.blocks", "", `{"object":"chat.completion.chunk","choices":[{"index":0,"message.blocks":{"text":"PIECE"}}]}`, []any{"choices", 0, "message.blocks", "text"}},
		{"responses shell stdout delta", "event: response.shell_call_output_content.delta\n", `{"type":"response.shell_call_output_content.delta","item_id":"sh_1","output_index":0,"command_index":0,"delta":{"stdout":"PIECE","stderr":""}}`, []any{"delta", "stdout"}},
		{"responses shell stderr delta", "event: response.shell_call_output_content.delta\n", `{"type":"response.shell_call_output_content.delta","item_id":"sh_1","output_index":0,"command_index":0,"delta":{"stdout":"","stderr":"PIECE"}}`, []any{"delta", "stderr"}},
		{"responses output_text.delta extension field", "event: response.output_text.delta\n", `{"type":"response.output_text.delta","item_id":"m_1","output_index":0,"content_index":0,"delta":"main","reasoning_content":"PIECE"}`, []any{"reasoning_content"}},
		{"responses audio.delta extension field", "event: response.audio.delta\n", `{"type":"response.audio.delta","delta":"QUJD","reasoning_content":"PIECE"}`, []any{"reasoning_content"}},
		{"responses output_audio.delta extension field", "event: response.output_audio.delta\n", `{"type":"response.output_audio.delta","delta":"QUJD","reasoning_content":"PIECE"}`, []any{"reasoning_content"}},
		{"native custom response.chunk", "", `{"type":"response.chunk","chunk":"PIECE"}`, []any{"chunk"}},
		{"native custom response.stream.generating", "", `{"type":"response.stream.generating","token":"PIECE"}`, []any{"token"}},
		{"native custom response.acme.delta", "", `{"type":"response.acme.delta","delta":{"text":"PIECE"}}`, []any{"delta", "text"}},
		{"native custom message_delta text", "", `{"type":"message_delta","text":"PIECE"}`, []any{"text"}},
		{"native object key 0 under choices", "", `{"choices":{"0":{"message":{"text":"PIECE"}}}}`, []any{"choices", "0", "message", "text"}},
		{"native empty parent key before responseId", "", `{"":{"responseId":"PIECE"}}`, []any{"", "responseId"}},
		{"anthropic thinking_delta", "event: content_block_delta\n", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"PIECE"}}`, []any{"delta", "thinking"}},
		{"anthropic compaction_delta", "event: content_block_delta\n", `{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"PIECE"}}`, []any{"delta", "content"}},
		{"legacy Completions", "", `{"object":"text_completion","choices":[{"index":0,"text":"PIECE"}]}`, []any{"choices", 0, "text"}},
		{"Gemini generic message event", "event: message\n", `{"candidates":[{"index":0,"content":{"parts":[{"text":"PIECE"}]}}]}`, []any{"candidates", 0, "content", "parts", 0, "text"}},
		{"native ADK text with usageMetadata", "", `{"content":{"parts":[{"text":"PIECE"}]},"usageMetadata":{"candidatesTokenCount":3}}`, []any{"content", "parts", 0, "text"}},
		{"native custom choices", "", `{"choices":[{"result":"PIECE"}]}`, []any{"choices", 0, "result"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			event := func(piece string) string {
				return tc.SSE + "data: " + strings.Replace(tc.body, "PIECE", piece, 1) + "\n\n"
			}
			got := throughWriter(event("dummy_"), event("standin"))
			var merged strings.Builder
			for _, line := range strings.Split(got, "\n") {
				if data, ok := strings.CutPrefix(line, "data: "); ok {
					var e any
					if err := json.Unmarshal([]byte(data), &e); err != nil {
						t.Fatal(err)
					}
					merged.WriteString(str(get(e, tc.path...)))
				}
			}
			if merged.String() != "real_secret" {
				t.Errorf("combined delta = %q, want the restored value", merged.String())
			}
		})
	}
	// Per SSE spec, "event: message", a colonless "event" line, and an omitted
	// event: line are the same default event and must continue the same stream.
	msg1 := "event: message\ndata: " + `{"candidates":[{"index":0,"content":{"parts":[{"text":"dummy_"}]}}]}` + "\n\n"
	msg2 := "data: " + `{"candidates":[{"index":0,"content":{"parts":[{"text":"standin"}]}}]}` + "\n\n"
	if got := throughWriter(msg1, msg2); !strings.Contains(got, `"text":"real_secret"`) {
		t.Errorf("event: message followed by omitted event: lost split placeholder: %s", got)
	}
	reset1 := "event: message \nevent\ndata: " + `{"candidates":[{"index":0,"content":{"parts":[{"text":"dummy_"}]}}]}` + "\n\n"
	reset2 := "event: message\ndata: " + `{"candidates":[{"index":0,"content":{"parts":[{"text":"standin"}]}}]}` + "\n\n"
	if got := throughWriter(reset1, reset2); !strings.Contains(got, `"text":"real_secret"`) {
		t.Errorf("colonless event reset line did not match event: message: %s", got)
	}
	// AI Studio Chat-compat marks only the opening <thought> chunk with
	// extra_content.google.thought: true; the closing chunk omits it and must
	// still continue the same text stream.
	thought1 := `data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<thought>dummy_","extra_content":{"google":{"thought":true}}}}]}` + "\n\n"
	thought2 := `data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"standin</thought>fin"}}]}` + "\n\n"
	if got := throughWriter(thought1, thought2); !strings.Contains(got, `real_secret</thought>fin`) {
		t.Errorf("Chat-compat marked thought split across unmarked chunk failed: %s", got)
	}
	// JSON-escaped item_id (\u0069 == "i") must share the same event context.
	esc1 := "event: response.output_text.delta\ndata: " + `{"type":"response.output_text.delta","item_id":"i","output_index":0,"content_index":0,"delta":"dummy_"}` + "\n\n"
	esc2 := "event: response.output_text.delta\ndata: " + `{"type":"response.output_text.delta","item_id":"\u0069","output_index":0,"content_index":0,"delta":"standin"}` + "\n\n"
	if got := throughWriter(esc1, esc2); !strings.Contains(got, `"delta":"real_secret"`) {
		t.Errorf("escaped item_id broke split restore: %s", got)
	}
	// Nested index under an empty key must not clobber top-level index.
	emp1 := "event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"dummy_"},"":{"index":5}}` + "\n\n"
	emp2 := "event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"standin"},"":{"index":6}}` + "\n\n"
	if got := throughWriter(emp1, emp2); !strings.Contains(got, `"text":"real_secret"`) {
		t.Errorf("nested index under empty key broke top-level context: %s", got)
	}
	// Gemini snake_case partial_args with will_continue and omitted id on chunk 2.
	part1 := `data: {"candidates":[{"index":0,"content":{"parts":[{"function_call":{"id":"call_1","name":"edit","partial_args":[{"json_path":"$.x","string_value":"dummy_","will_continue":true}]}}]}}]}` + "\n\n"
	part2 := `data: {"candidates":[{"index":0,"content":{"parts":[{"function_call":{"partial_args":[{"json_path":"$.x","string_value":"standin"}]}}]}}]}` + "\n\n"
	if got := throughWriter(part1, part2); !strings.Contains(got, `"string_value":"real_secret"`) {
		t.Errorf("snake_case partial_args continuation lost split placeholder: %s", got)
	}
	// Case-different extension key ("Type") must never override top-level "type"
	// (neither breaking asJSON escaping nor triggering waitEvents "ping").
	withQuote := func(t *testing.T) string {
		t.Helper()
		return placeholder("SECRET", `a"b\c`)
	}
	qph := withQuote(t)
	q1 := `data: {"type":"response.function_call_arguments.delta","Type":"caption","item_id":"fc_1","output_index":0,"delta":"{\"v\":\"` + qph[:6] + `"}` + "\n\n"
	q2 := `data: {"type":"response.function_call_arguments.delta","Type":"ping","item_id":"fc_1","output_index":0,"delta":"` + qph[6:] + `\"}"}` + "\n\n"
	gotQ := throughWriter(q1, q2)
	var argsBuf strings.Builder
	for _, line := range strings.Split(gotQ, "\n") {
		if d, ok := strings.CutPrefix(line, "data: "); ok {
			var ev map[string]any
			if err := json.Unmarshal([]byte(d), &ev); err != nil {
				t.Fatalf("case-key event invalid JSON: %v", err)
			}
			argsBuf.WriteString(str(ev["delta"]))
		}
	}
	var parsedArgs map[string]string
	if err := json.Unmarshal([]byte(argsBuf.String()), &parsedArgs); err != nil || parsedArgs["v"] != `a"b\c` {
		t.Errorf("case-different Type broke function_call_arguments JSON: %q (%v)", argsBuf.String(), err)
	}
	// Equivalent JSON numeric spellings (1.0/1e0, 0e65/0, 0.00..01e65/1) must share the same index.
	num1 := `data: {"type":"content_block_delta","index":1.0,"delta":{"type":"text_delta","text":"dummy_"}}` + "\n\n"
	num2 := `data: {"type":"content_block_delta","index":1e0,"delta":{"type":"text_delta","text":"standin"}}` + "\n\n"
	if got := throughWriter(num1, num2); !strings.Contains(got, `"text":"real_secret"`) {
		t.Errorf("equivalent numeric index spellings broke continuation: %s", got)
	}
	zero1 := `data: {"type":"content_block_delta","index":0e9223372036854775808,"delta":{"type":"text_delta","text":"dummy_"}}` + "\n\n"
	zero2 := `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"standin"}}` + "\n\n"
	if got := throughWriter(zero1, zero2); !strings.Contains(got, `"text":"real_secret"`) {
		t.Errorf("zero numeric index spellings (huge exp vs 0) broke continuation: %s", got)
	}
	one1 := `data: {"type":"content_block_delta","index":0.` + strings.Repeat("0", 64) + `1e65,"delta":{"type":"text_delta","text":"dummy_"}}` + "\n\n"
	one2 := `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"standin"}}` + "\n\n"
	if got := throughWriter(one1, one2); !strings.Contains(got, `"text":"real_secret"`) {
		t.Errorf("scaled-one numeric index spelling broke continuation: %s", got)
	}
	// A failed downstream Write on a truncated JSON frame must keep the held
	// frame intact until Finish.
	fwTrunc := &failOnceWriter{ResponseRecorder: httptest.NewRecorder(), fail: true}
	fwTrunc.Header().Set("Content-Type", "text/event-stream")
	wt := NewWriter(fwTrunc)
	wt.WriteHeader(http.StatusOK)
	tf1 := `data: {"text":"word"}` + "\n\n"
	tf2 := `data: {"text":` + "\n\n"
	if _, err := wt.Write([]byte(tf1)); err != nil {
		t.Fatalf("first frame write: %v", err)
	}
	if _, err := wt.Write([]byte(tf2)); err == nil {
		t.Fatal("expected error from downstream writer on truncated frame")
	}
	wt.Finish()
	if fwTrunc.Body.String() != tf1 {
		t.Errorf("after failed release on truncated frame, Finish wrote %q, want %q", fwTrunc.Body.String(), tf1)
	}
	// Chat tool_calls with null index keep matching even if id appears late or
	// is omitted on the middle chunk, while call-a / call-b (with index:0) /
	// omitted-id chunks stay isolated.
	tc0 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":null,"id":"call-a","function":{"arguments":"open:"}}]}}]}` + "\n\n"
	tc1 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":null,"function":{"arguments":"dummy_"}}]}}]} ` + "\n\n"
	tc2 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":null,"id":"call-a","function":{"arguments":"standin"}}]}}]} ` + "\n\n"
	if got := throughWriter(tc0, tc1, tc2); !strings.Contains(got, `"arguments":"real_secret"`) {
		t.Errorf("tool_calls continuation with null index lost split placeholder: %s", got)
	}
	tcLate1 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":null,"function":{"arguments":"dummy_"}}]}}]} ` + "\n\n"
	tcLate2 := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":null,"id":"call-a","function":{"arguments":"standin"}}]}}]} ` + "\n\n"
	if got := throughWriter(tcLate1, tcLate2); !strings.Contains(got, `"arguments":"real_secret"`) {
		t.Errorf("tool_calls with late id and null index lost split placeholder: %s", got)
	}
	// Reordered reasoning_details by id omitting type, and ambiguous dual
	// index:0 continuations, both match origin/main cleanly.
	rdMove1 := `data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"id":"r1","type":"reasoning.text","text":"dummy_"},{"id":"r2","text":"x"}]}}]}` + "\n\n"
	rdMove2 := `data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"id":"r2","text":"y"},{"id":"r1","text":"standin"}]}}]}` + "\n\n"
	if got := throughWriter(rdMove1, rdMove2); !strings.Contains(got, `"text":"real_secret"`) {
		t.Errorf("reordered reasoning_details omitting type lost split placeholder: %s", got)
	}
	rdAmb1 := `data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"index":0,"text":"word"}]}}]}` + "\n\n"
	rdAmb2 := `data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"RIGHT"},{"index":0,"text":"LEFT"}]}}]}` + "\n\n"
	if got := throughWriter(rdAmb1, rdAmb2); got != rdAmb1+rdAmb2 {
		t.Errorf("ambiguous dual index:0 reasoning_details moved tail:\n got %s\nwant %s", got, rdAmb1+rdAmb2)
	}
	// A failed downstream Write while flushing a held frame must neither drop
	// the held frame before Finish nor inflate Put counts for the unwritten frame.
	fw := &failOnceWriter{ResponseRecorder: httptest.NewRecorder(), fail: true}
	fw.Header().Set("Content-Type", "text/event-stream")
	w := NewWriter(fw)
	w.WriteHeader(http.StatusOK)
	f1 := `data: {"choices":[{"index":0,"delta":{"content":"word"}}]}` + "\n\n"
	f2 := `data: {"choices":[{"index":0,"delta":{"content":"dummy_standin"}}]}` + "\n\n"
	if _, err := w.Write([]byte(f1)); err != nil {
		t.Fatalf("first frame write: %v", err)
	}
	if _, err := w.Write([]byte(f2)); err == nil {
		t.Fatal("expected error from downstream writer")
	}
	w.Finish()
	if fw.Body.String() != f1 {
		t.Errorf("after failed flush, Finish wrote %q, want %q", fw.Body.String(), f1)
	}
	if len(w.Put()) != 0 {
		t.Errorf("after failed flush, Put() = %v, want empty", w.Put())
	}
	// If downstream already accepted the bytes (returned len(p), err), Finish
	// must not emit the held frame a second time.
	fwWritten := &failAfterWriteWriter{ResponseRecorder: httptest.NewRecorder(), fail: true}
	fwWritten.Header().Set("Content-Type", "text/event-stream")
	ww := NewWriter(fwWritten)
	ww.WriteHeader(http.StatusOK)
	if _, err := ww.Write([]byte(f1)); err != nil {
		t.Fatalf("first frame write: %v", err)
	}
	if _, err := ww.Write([]byte("event: done\ndata:\n\n")); err == nil {
		t.Fatal("expected error from downstream writer")
	}
	ww.Finish()
	if fwWritten.Body.String() != f1 {
		t.Errorf("after written-with-err release, Finish duplicated frame: got %q, want %q", fwWritten.Body.String(), f1)
	}
	// If downstream accepts only the SSE header line (0 < n < len(p)) before
	// returning an error, Finish must still emit the held event so the data
	// line isn't lost.
	fwPart := &failPartialWriter{ResponseRecorder: httptest.NewRecorder(), limit: 6}
	fwPart.Header().Set("Content-Type", "text/event-stream")
	wp := NewWriter(fwPart)
	wp.WriteHeader(http.StatusOK)
	pf1 := "id: 1\ndata: " + `{"text":"word"}` + "\n\n"
	pf2 := `data: {"text":"next"}` + "\n\n"
	if _, err := wp.Write([]byte(pf1)); err != nil {
		t.Fatalf("first frame write: %v", err)
	}
	if _, err := wp.Write([]byte(pf2)); err == nil {
		t.Fatal("expected partial write error from downstream writer")
	}
	wp.Finish()
	if !strings.Contains(fwPart.Body.String(), `data: {"text":"word"}`) {
		t.Errorf("after partial header write error, Finish lost event data: %q", fwPart.Body.String())
	}
	// String choice index "0" followed by numeric 0 (reordered after "1") must
	// still restore a split placeholder.
	strIdx1 := `data: {"choices":[{"index":"0","delta":{"content":"dummy_"}}]}` + "\n\n"
	strIdx2 := `data: {"choices":[{"index":"1","delta":{"content":"x"}},{"index":0,"delta":{"content":"standin"}}]}` + "\n\n"
	if got := throughWriter(strIdx1, strIdx2); !strings.Contains(got, `"content":"real_secret"`) {
		t.Errorf("string-to-numeric choice index broke split restore: %s", got)
	}
	// Responses reasoning_summary_text.delta with summary_index:0 followed by
	// omitted summary_index (default 0) must still restore a split placeholder.
	sum1 := "event: response.reasoning_summary_text.delta\ndata: " + `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"dummy_"}` + "\n\n"
	sum2 := "event: response.reasoning_summary_text.delta\ndata: " + `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"delta":"standin"}` + "\n\n"
	if got := throughWriter(sum1, sum2); !strings.Contains(got, `"delta":"real_secret"`) {
		t.Errorf("omitted default summary_index broke split restore: %s", got)
	}
	// Short write on release() missing only the final \n of \r\n\r\n must not
	// duplicate the event on Finish().
	crlf1 := "data: " + `{"text":"word"}` + "\r\n\r\n"
	fwShort := &failPartialWriter{ResponseRecorder: httptest.NewRecorder(), limit: len(crlf1) - 1}
	fwShort.Header().Set("Content-Type", "text/event-stream")
	ws := NewWriter(fwShort)
	ws.WriteHeader(http.StatusOK)
	if _, err := ws.Write([]byte(crlf1)); err != nil {
		t.Fatalf("first frame write: %v", err)
	}
	if _, err := ws.Write([]byte("event: done\ndata:\n\n")); err == nil {
		t.Fatal("expected short write error on release")
	}
	ws.Finish()
	if strings.Count(fwShort.Body.String(), `"text":"word"`) != 1 {
		t.Errorf("short write on release duplicated event on Finish: %q", fwShort.Body.String())
	}
}

type failPartialWriter struct {
	*httptest.ResponseRecorder
	limit int
}

func (f *failPartialWriter) Write(p []byte) (int, error) {
	if f.limit > 0 {
		n := min(f.limit, len(p))
		f.limit = 0
		_, _ = f.ResponseRecorder.Write(p[:n])
		return n, errors.New("partial write failed")
	}
	return f.ResponseRecorder.Write(p)
}

type failOnceWriter struct {
	*httptest.ResponseRecorder
	fail bool
}

func (f *failOnceWriter) Write(p []byte) (int, error) {
	if f.fail {
		f.fail = false
		return 0, errors.New("write failed")
	}
	return f.ResponseRecorder.Write(p)
}

type failAfterWriteWriter struct {
	*httptest.ResponseRecorder
	fail bool
}

func (f *failAfterWriteWriter) Write(p []byte) (int, error) {
	n, _ := f.ResponseRecorder.Write(p)
	if f.fail {
		f.fail = false
		return n, errors.New("write failed after accepting bytes")
	}
	return n, nil
}

func TestWriterRestoresDeltasAfterCandidateReordering(t *testing.T) {
	withStandIn(t)
	for _, tc := range []struct{ name, initial, reordered, restored string }{
		{
			"Chat choices",
			`{"choices":[{"index":0,"delta":{"content":"dummy_"}},{"index":1,"delta":{"content":"other"}}]}`,
			`{"choices":[{"index":1,"delta":{"content":"other"}},{"index":0,"delta":{"content":"standin"}}]}`,
			`{"choices":[{"index":1,"delta":{"content":"other"}},{"index":0,"delta":{"content":"real_secret"}}]}`,
		},
		{
			"Gemini candidates",
			`{"candidates":[{"index":0,"content":{"parts":[{"text":"dummy_"}]}},{"index":1,"content":{"parts":[{"text":"other"}]}}]}`,
			`{"candidates":[{"index":1,"content":{"parts":[{"text":"other"}]}},{"index":0,"content":{"parts":[{"text":"standin"}]}}]}`,
			`{"candidates":[{"index":1,"content":{"parts":[{"text":"other"}]}},{"index":0,"content":{"parts":[{"text":"real_secret"}]}}]}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			frame := func(data string) string { return "data: " + data + "\n\n" }
			want := frame(strings.Replace(tc.initial, "dummy_", "", 1)) + frame(tc.restored)
			if got := throughWriter(frame(tc.initial), frame(tc.reordered)); got != want {
				t.Errorf("reordered stream:\n got %s\nwant %s", got, want)
			}
		})
	}
}

// A stream's protocol, as a test reads it back: each string of an event
// under the text it is a piece of, as the protocol says which that is.
type protocol struct {
	name  string
	event func(text, piece string) string
	// texts puts each piece of an event's JSON under its text
	texts func(data map[string]any, n int, into map[string]*strings.Builder)
}

func add(into map[string]*strings.Builder, k, s string) {
	if into[k] == nil {
		into[k] = &strings.Builder{}
	}
	into[k].WriteString(s)
}

func get(v any, path ...any) any {
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, _ := v.(map[string]any)
			v = m[k]
		case int:
			a, _ := v.([]any)
			if k >= len(a) {
				return nil
			}
			v = a[k]
		}
	}
	return v
}

func str(v any) string { s, _ := v.(string); return s }

var protocols = []protocol{
	{
		// Chat Completions: one call's pieces by its index; a call's
		// first piece sits at tool_calls.0 as the call before it did
		name: "chat tool_calls",
		event: func(text, piece string) string {
			// in the order OpenAI writes a chunk's keys: the choice's
			// index before its delta
			a, _ := json.Marshal(piece)
			return `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":` + text +
				`,"id":"call_` + text + `","type":"function","function":{"name":"read_file","arguments":` + string(a) + `}}]}}]}` + "\n\n"
		},
		texts: func(d map[string]any, _ int, into map[string]*strings.Builder) {
			c := get(d, "choices", 0, "delta", "tool_calls", 0)
			add(into, fmt.Sprint(get(c, "index")), str(get(c, "function", "arguments")))
		},
	},
	{
		// a vendor's Chat Completions that names a call by its id alone
		name: "chat tool_calls by id",
		event: func(text, piece string) string {
			a, _ := json.Marshal(piece)
			return `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_` + text +
				`","type":"function","function":{"name":"read_file","arguments":` + string(a) + `}}]}}]}` + "\n\n"
		},
		texts: func(d map[string]any, _ int, into map[string]*strings.Builder) {
			c := get(d, "choices", 0, "delta", "tool_calls", 0)
			add(into, str(get(c, "id")), str(get(c, "function", "arguments")))
		},
	},
	{
		name: "responses function_call_arguments.delta",
		event: func(text, piece string) string {
			b, _ := json.Marshal(map[string]any{"type": "response.function_call_arguments.delta", "item_id": "fc_" + text, "output_index": json.Number(text), "delta": piece})
			return "event: response.function_call_arguments.delta\ndata: " + string(b) + "\n\n"
		},
		texts: func(d map[string]any, _ int, into map[string]*strings.Builder) {
			add(into, str(d["item_id"]), str(d["delta"]))
		},
	},
	{
		name: "anthropic input_json_delta",
		event: func(text, piece string) string {
			b, _ := json.Marshal(map[string]any{"type": "content_block_delta", "index": json.Number(text), "delta": map[string]any{"type": "input_json_delta", "partial_json": piece}})
			return "event: content_block_delta\ndata: " + string(b) + "\n\n"
		},
		texts: func(d map[string]any, _ int, into map[string]*strings.Builder) {
			add(into, fmt.Sprint(d["index"]), str(get(d, "delta", "partial_json")))
		},
	},
	{
		// Gemini: the thoughts, then the answer, both at parts.0.text
		name: "gemini thought and text",
		event: func(text, piece string) string {
			part := map[string]any{"text": piece}
			if text == "0" {
				part["thought"] = true
			}
			b, _ := json.Marshal(map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"role": "model", "parts": []any{part}}, "index": 0}}})
			return "data: " + string(b) + "\n\n"
		},
		texts: func(d map[string]any, _ int, into map[string]*strings.Builder) {
			p := get(d, "candidates", 0, "content", "parts", 0)
			k := "text"
			if get(p, "thought") == true {
				k = "thought"
			}
			add(into, k, str(get(p, "text")))
		},
	},
	{
		// Gemini's streamed arguments (streamFunctionCallArguments): a
		// value's pieces by its jsonPath while willContinue; each call
		// here has one, of the same name, and its last piece ends it
		name: "gemini partialArgs",
		event: func(text, piece string) string {
			arg := map[string]any{"jsonPath": "$.Path", "stringValue": piece}
			if strings.HasSuffix(text, "+") {
				arg["willContinue"] = true
			}
			b, _ := json.Marshal(map[string]any{"candidates": []any{map[string]any{"content": map[string]any{"role": "model", "parts": []any{
				map[string]any{"functionCall": map[string]any{"name": "read_file", "partialArgs": []any{arg}, "willContinue": true}}}}, "index": 0}}})
			return "data: " + string(b) + "\n\n"
		},
		texts: func(d map[string]any, n int, into map[string]*strings.Builder) {
			a := get(d, "candidates", 0, "content", "parts", 0, "functionCall", "partialArgs", 0)
			add(into, fmt.Sprint(get(a, "jsonPath"), "#", into["#calls"].Len()), str(get(a, "stringValue")))
			if get(a, "willContinue") != true {
				add(into, "#calls", "x") // the next piece is a call of its own
			}
		},
	},
}

// readBack puts each piece of a stream under its text, and counts the
// events.
func readBack(t *testing.T, p protocol, stream string) (map[string]string, int) {
	t.Helper()
	into := map[string]*strings.Builder{"#calls": {}}
	n := 0
	for _, ev := range strings.Split(stream, "\n\n") {
		_, d, ok := strings.Cut(ev, "data: ")
		if !ok {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(d), &m); err != nil {
			t.Fatalf("%s: event %q: %v", p.name, d, err)
		}
		p.texts(m, n, into)
		n++
	}
	out := map[string]string{}
	for k, b := range into {
		if k != "#calls" {
			out[k] = b.String()
		}
	}
	return out, n
}

// Each text a stream holds a tail of goes on with that tail only: never
// with another tool call's, another block's, or the thoughts' before the
// answer (#7837). In every protocol the Writer reads, a text ending in a
// stand-in's first letter is followed by another text's first piece, and
// each text the agent puts together is the one the vendor sent.
func TestWriterTailStaysWithItsText(t *testing.T) {
	withStandIn(t)
	for _, p := range protocols {
		var evs []string
		texts := [][2]string{{"0", `{"Path":"/tmp/foo1.md`}, {"1", `{"Path":"/tmp/foo2.md`}, {"1", `"}`}, {"0", `"}`}}
		if p.name == "gemini thought and text" {
			texts = [][2]string{{"0", "I should read"}, {"1", "Here it is, read"}, {"1", " and done"}}
		}
		if p.name == "gemini partialArgs" {
			texts = [][2]string{{"0", "/tmp/foo1.md"}, {"1+", "/tmp/foo2.md"}, {"1", ""}, {"2", "/tmp/foo3.md"}}
		}
		for _, x := range texts {
			evs = append(evs, p.event(x[0], x[1]))
		}
		in := strings.Join(evs, "")
		out := throughWriter(evs...)
		want, _ := readBack(t, p, in)
		got, n := readBack(t, p, out)
		if n != len(evs) || fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("%s: texts changed\n got %q\nwant %q\n%s", p.name, got, want, out)
		}
	}
}

// A placeholder split between two pieces of one text is still put back
// whole, with another text's piece, ending in what may be a placeholder's
// start, just before it.
func TestWriterSplitPlaceholderBesideAnotherText(t *testing.T) {
	withStandIn(t)
	const v = "Hunter2Charlie-99"
	ph := placeholder("SECRET", v)
	for _, p := range protocols {
		other, mine, rest := "0", "1", "1"
		text := `{"Path":"/tmp/` + ph + `/notes.md"}`
		before := `{"Path":"/tmp/foo1.md`
		switch p.name {
		case "gemini thought and text":
			text, before = "the key is "+ph+", read", "I should read"
		case "gemini partialArgs":
			mine, text, before = "1+", "/tmp/"+ph+"/notes.md", "/tmp/foo1.md"
		}
		at := strings.Index(text, ph)
		for cut := at + 1; cut < at+len(ph); cut++ {
			evs := []string{p.event(other, before), p.event(mine, text[:cut]), p.event(rest, text[cut:])}
			got, n := readBack(t, p, throughWriter(evs...))
			want, _ := readBack(t, p, strings.Join(evs, ""))
			for k := range want {
				want[k] = strings.ReplaceAll(want[k], ph, v)
			}
			if n != len(evs) || fmt.Sprint(got) != fmt.Sprint(want) {
				t.Fatalf("%s cut at %d:\n got %q\nwant %q", p.name, cut, got, want)
			}
		}
	}
}
