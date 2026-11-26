package redact

import (
	"bytes"
	"encoding/json"
	"net/http"
	"slices"
	"strconv"
	"strings"
)

// Writer puts the values back in what a vendor answers, on its way to the
// agent. A stream is put back event by event, as it comes; anything else
// when it is done, at Finish.
//
// A placeholder a stream splits between two events — {{API_ in one text
// delta, KEY_k3v9x2mq}} in the next — is put back whole: an event whose
// text ends in what may be the start of one is held until the next, and if
// that one goes on with the same text the start moves over to it. Nothing
// is added or dropped, so the text the agent puts together is the same.
// The same text is the same string of the stream, not one at the same
// place in the next event: another tool call's arguments, Gemini's next
// whole functionCall or its answer after its thoughts are texts of their
// own (pieces), and a start moved onto one of them took its last letter
// from one path and gave it to the next (#7837).
type Writer struct {
	w       http.ResponseWriter
	mode    int // 0 not yet known, 1 a stream, 2 a whole body, 3 left as it is
	buf     bytes.Buffer
	pending *held
	done    bool
	put     map[string]int // placeholders put back, each place counted
}

const (
	unknown = iota
	stream
	whole
	asIs
)

// heldTail is one string's possible placeholder start held back from an event.
type heldTail struct {
	key    string   // canonical text identity key
	altIDs []string // secondary explicit ids on indexed array elements along the path
	tail   string
	ord    int    // string ordinal in walk order
	cut    string // what is left without the tail, put back
}

// held is an event one or more of whose texts ended in what may be the
// start of a placeholder: each one's tail by which text it is a piece of.
type held struct {
	ctx       string
	tails     []heldTail
	full      []byte // the event as it is
	pre, post []byte
	data      []byte // the event's JSON, put back
}

// out is the held event as it goes out ahead of the next one: cut where
// the next event goes on with that text, whose tail it carries, and
// whole everywhere else, where the text ended.
func (h *held) out(used []bool) []byte {
	if len(used) == 0 {
		return h.full
	}
	var cuts map[int]string
	for i, ht := range h.tails {
		if used[i] {
			if cuts == nil {
				cuts = map[int]string{}
			}
			cuts[ht.ord] = ht.cut
		}
	}
	if len(cuts) == 0 {
		return h.full
	}
	idx := 0
	d, _ := walk(h.data, func(_, _, s string) string {
		i := idx
		idx++
		if c, ok := cuts[i]; ok {
			return c
		}
		return s
	})
	return rebuild(h.pre, d, h.post)
}

// NewWriter wraps w; Finish must be called when the handler is done.
func NewWriter(w http.ResponseWriter) *Writer { return &Writer{w: w} }

func (rw *Writer) Header() http.Header { return rw.w.Header() }

// Put is how many places of each placeholder the answer had put back,
// so far: for the record of what masking did.
func (rw *Writer) Put() map[string]int { return rw.put }

func (rw *Writer) count() map[string]int {
	if rw.put == nil {
		rw.put = map[string]int{}
	}
	return rw.put
}

// Unwrap is for http.ResponseController.
func (rw *Writer) Unwrap() http.ResponseWriter { return rw.w }

func (rw *Writer) decide(first []byte) {
	h := rw.w.Header()
	ct := h.Get("Content-Type")
	switch {
	case h.Get("Content-Encoding") != "" && !strings.EqualFold(h.Get("Content-Encoding"), "identity"):
		rw.mode = asIs
	case strings.Contains(ct, "event-stream"):
		rw.mode = stream
	case ct == "" && (bytes.HasPrefix(first, []byte("event:")) || bytes.HasPrefix(first, []byte("data:")) || bytes.HasPrefix(first, []byte(":"))):
		rw.mode = stream // the ChatGPT backend streams without saying so
	case binaryType(ct):
		rw.mode = asIs // speech, an image, a video: no text to put back
	default:
		rw.mode = whole
	}
}

// binaryType says a Content-Type is of bytes that aren't text: audio, an
// image, a video, a file. Held whole, speech played only when it ended.
func binaryType(ct string) bool {
	ct = strings.ToLower(ct)
	for _, t := range []string{"audio/", "image/", "video/", "application/octet-stream", "application/pdf", "application/zip"} {
		if strings.HasPrefix(ct, t) {
			return true
		}
	}
	return false
}

func (rw *Writer) WriteHeader(code int) {
	if rw.mode == unknown {
		h := rw.w.Header()
		if ct := h.Get("Content-Type"); ct != "" {
			rw.decide(nil)
		}
		// what is put back is not as long as what came
		if rw.mode != asIs {
			h.Del("Content-Length")
		}
	}
	rw.w.WriteHeader(code)
}

func (rw *Writer) Write(p []byte) (int, error) {
	if rw.mode == unknown {
		rw.decide(p)
		if rw.mode != asIs {
			rw.w.Header().Del("Content-Length")
		}
	}
	switch rw.mode {
	case asIs:
		return rw.w.Write(p)
	case whole:
		return rw.buf.Write(p)
	}
	rw.buf.Write(p)
	for {
		b := rw.buf.Bytes()
		// a comment between two events (": keepalive") goes on as it
		// comes: held for the event after it, an agent behind a proxy
		// heard nothing while its vendor reasoned silently, and the
		// proxy hung up (#2402). It carries nothing to put back, and an
		// event held for its placeholder's tail goes on after it as well.
		if c := commentLines(b); c > 0 {
			if _, err := rw.w.Write(b[:c]); err != nil {
				return len(p), err
			}
			rw.buf.Next(c)
			continue
		}
		i, n := eventEnd(b)
		if i < 0 {
			break
		}
		ev := append([]byte(nil), b[:i+n]...)
		rw.buf.Next(i + n)
		if err := rw.event(ev); err != nil {
			return len(p), err
		}
	}
	return len(p), nil
}

// commentLines is how long the whole comment lines (": keepalive") and
// blank lines at the start of b are: b starts where an event does, so none
// of them is part of one.
func commentLines(b []byte) int {
	n := 0
	for n < len(b) && (b[n] == ':' || b[n] == '\n' || b[n] == '\r') {
		i := bytes.IndexByte(b[n:], '\n')
		if i < 0 || b[n] == '\r' && i != 1 {
			break
		}
		n += i + 1
	}
	return n
}

// eventEnd is where the first complete event in b ends, and the length of
// the blank line that ends it.
func eventEnd(b []byte) (int, int) {
	i := bytes.Index(b, []byte("\n\n"))
	j := bytes.Index(b, []byte("\r\n\r\n"))
	switch {
	case i < 0 && j < 0:
		return -1, 0
	case j >= 0 && (i < 0 || j < i):
		return j, 4
	default:
		return i, 2
	}
}

func (rw *Writer) Flush() {
	if f, ok := rw.w.(http.Flusher); ok && rw.mode != whole {
		f.Flush()
	}
}

// Finish writes what is still held.
func (rw *Writer) Finish() {
	if rw.done {
		return
	}
	rw.done = true
	switch rw.mode {
	case whole:
		rw.w.Write(restoreJSON(rw.buf.Bytes(), rw.count()))
	case stream:
		if rw.pending != nil {
			rw.w.Write(rw.pending.full)
			rw.pending = nil
		}
		if rw.buf.Len() > 0 {
			rw.w.Write([]byte(restore(rw.buf.String(), false, rw.count())))
		}
	}
	rw.Flush()
}

// event puts the values back in one event and writes it, or holds it.
//
// Every text of it is put back but what a vendor sealed (a signature, a
// file's data): a tool's own arguments, as Gemini streams them whole, use
// the keys a vendor uses for itself (name, type, id, url, status), and a
// name the model wrote under "name" kept its placeholder. Only a text is
// held for its tail, never an id or Responses' random obfuscation, and
// each text's tail is held for itself: obfuscation that ended in a kind's
// first letters took the hold from the delta beside it, and the next
// event's half of that placeholder went out as it came.
func (rw *Writer) event(ev []byte) error {
	if rw.pending == nil && !holding() {
		_, err := rw.w.Write(ev) // nothing masked to put back
		return err
	}
	name, data, pre, post, ok := splitEvent(ev)
	if !ok {
		if err := rw.release(); err != nil {
			return err
		}
		_, err := rw.w.Write([]byte(restore(string(ev), false, rw.count())))
		return err
	}
	typ, ctx, ok := topFields(name, data)
	if !ok {
		if p := rw.pending; p != nil {
			if _, err := rw.w.Write(p.full); err != nil {
				return err
			}
			rw.pending = nil
		}
		_, err := rw.w.Write(ev)
		return err
	}
	if rw.pending != nil && slices.Contains(waitEvents, typ) {
		// a keepalive between two pieces of a text goes on ahead of the
		// piece held: released for it, the held piece went out with the
		// start of its placeholder as it was, the next piece with the
		// rest, and neither was a placeholder to put back (#5287)
		_, err := rw.w.Write(rebuild(pre, restoreJSON(data, rw.count()), post))
		return err
	}
	var texts *pieces // read once, if this event goes on with a text or holds one
	// the starts of placeholders the last event ended in go on here, each
	// on the string of the text it was cut from
	var carry map[int]string
	if p := rw.pending; p != nil {
		var used []bool
		if p.ctx == ctx {
			texts = piecesOf(data, typ)
			carry, used = texts.on(p.tails)
		}
		if _, err := rw.w.Write(p.out(used)); err != nil {
			return err
		}
		rw.pending = nil
	}
	var tails []heldTail
	var seenTail map[string]bool
	idx := 0
	restored, _ := walk(data, func(path, key, s string) string {
		i := idx
		idx++
		if c, ok := carry[i]; ok {
			s = c + s
		}
		if sealedText(key, s) {
			return s
		}
		// the tail is looked for before anything is put back, so a
		// placeholder whose last brace is still to come isn't put back
		// without it and the brace left behind
		if !kept[key] && key != "obfuscation" {
			if n := partialTail(s); n > 0 {
				if texts == nil {
					texts = piecesOf(data, typ)
				}
				// a string that is whole, or the last piece of its
				// text, goes out whole: nothing of the next event is
				// its own; so does a second string of one text
				if tkey, altIDs, _, more := texts.of(i); more && !seenTail[tkey] {
					if seenTail == nil {
						seenTail = map[string]bool{}
					}
					seenTail[tkey] = true
					tails = append(tails, heldTail{
						key:    tkey,
						altIDs: altIDs,
						tail:   s[len(s)-n:],
						ord:    i,
						cut:    Restore(s[:len(s)-n], asJSON(key, typ)),
					})
				}
			}
		}
		// counted here only: cut is s again without its tail, and
		// whichever of the two goes out, the places are these
		return restore(s, asJSON(key, typ), rw.count())
	})
	full := rebuild(pre, restored, post)
	if len(tails) == 0 {
		_, err := rw.w.Write(full)
		return err
	}
	rw.pending = &held{ctx: ctx, tails: tails, full: full, pre: pre, post: post, data: restored}
	return nil
}

// waitEvents are the events a stream sends while the model is slow, which
// say nothing of the answer: Anthropic's ping, which its API and magpie's
// encoder send in the middle of a tool call's arguments, Responses'
// keepalive, and response.in_progress and response.queued, magpie's
// keepalive on that protocol.
var waitEvents = []string{"ping", "keepalive", "response.in_progress", "response.queued"}

// sealedText says a string of a vendor's answer goes on as it came: what the
// vendor sealed (signatureKeys) and a file's encoded data.
func sealedText(key, s string) bool {
	return slices.Contains(signatureKeys, key) || strings.HasPrefix(s, "data:") ||
		key == "data" && len(s) >= 64 && !strings.ContainsAny(s, " \n")
}

// holding says some value is masked, and may be put back.
func holding() bool {
	mu.RLock()
	defer mu.RUnlock()
	return len(values) > 0
}

func (rw *Writer) release() error {
	if p := rw.pending; p != nil {
		rw.pending = nil
		_, err := rw.w.Write(p.full)
		return err
	}
	return nil
}

func rebuild(pre, data, post []byte) []byte {
	out := make([]byte, 0, len(pre)+len(data)+len(post))
	return append(append(append(out, pre...), data...), post...)
}

// splitEvent finds the one data line of an event that has JSON in it:
// what comes before the JSON, the JSON, and what comes after it.
func splitEvent(ev []byte) (name string, data, pre, post []byte, ok bool) {
	lines := bytes.SplitAfter(ev, []byte("\n"))
	at := -1
	off := 0
	for i, l := range lines {
		t := bytes.TrimRight(l, "\r\n")
		switch {
		case bytes.Equal(t, []byte("event")):
			name = ""
		case bytes.HasPrefix(t, []byte("event:")):
			v := t[6:]
			if len(v) > 0 && v[0] == ' ' {
				v = v[1:]
			}
			name = string(v)
		case bytes.HasPrefix(t, []byte("data:")):
			if at >= 0 {
				return "", nil, nil, nil, false // data over more than one line
			}
			at = i
		}
		if at < 0 {
			off += len(l)
		}
	}
	if at < 0 {
		return "", nil, nil, nil, false
	}
	l := lines[at]
	t := bytes.TrimRight(l, "\r\n")
	start := 5
	if len(t) > 5 && t[5] == ' ' {
		start = 6
	}
	d := t[start:]
	if len(d) == 0 || d[0] != '{' && d[0] != '[' {
		return "", nil, nil, nil, false
	}
	return name, d, ev[:off+start], ev[off+len(t):], true
}

// rawSlice borrows the raw JSON bytes of a top-level value during Unmarshal
// without copying large nested payloads (such as a 2 MB partial_image_b64).
type rawSlice []byte

func (r *rawSlice) UnmarshalJSON(b []byte) error {
	*r = b
	return nil
}

// normNum puts a valid JSON number into a canonical decimal-exponent form
// (e.g. 1, 1.0, 1e0, 0.00...01e65 -> "1"; -0.0, 0e99999999999999999999 -> "0")
// in O(len(s)).
func normNum(s string) string {
	neg := false
	rest := s
	if strings.HasPrefix(rest, "-") {
		neg = true
		rest = rest[1:]
	}
	mant, expStr, hasExp := rest, "", false
	if i := strings.IndexAny(rest, "eE"); i >= 0 {
		mant, expStr, hasExp = rest[:i], rest[i+1:], true
	}
	intPart, fracPart, hasDot := strings.Cut(mant, ".")
	digits := intPart
	if hasDot {
		digits = intPart + fracPart
	}
	digits = strings.TrimLeft(digits, "0")
	if digits == "" {
		return "0"
	}
	var exp int64
	if hasExp {
		var err error
		exp, err = strconv.ParseInt(expStr, 10, 64)
		if err != nil {
			return s
		}
	}
	trimmed := strings.TrimRight(digits, "0")
	scaleShift := int64(len(digits)-len(trimmed)) - int64(len(fracPart))
	// guard int64 addition against overflow on extreme exponents
	if (scaleShift > 0 && exp > (1<<62)-scaleShift) || (scaleShift < 0 && exp < -(1<<62)-scaleShift) {
		return s
	}
	scale := exp + scaleShift
	var b strings.Builder
	if neg {
		b.WriteByte('-')
	}
	b.WriteString(trimmed)
	if scale != 0 {
		b.WriteByte('e')
		b.WriteString(strconv.FormatInt(scale, 10))
	}
	return b.String()
}

func rawScalar(raw rawSlice) string {
	if len(raw) == 0 {
		return ""
	}
	switch raw[0] {
	case '"':
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return "s:" + s
		}
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		return "n:" + normNum(string(raw))
	}
	return ""
}

// topFields reads an event's effective type and top-level context (exact
// case-sensitive depth-1 keys only, never nested choice or candidate indices).
// Per SSE, an omitted event: line and "event: message" are the same default
// event; all other explicit event names stay distinct. ok is false if data
// isn't valid JSON.
func topFields(name string, data []byte) (typ, ctx string, ok bool) {
	var top map[string]rawSlice
	if data[0] == '[' {
		if !json.Valid(data) {
			return "", "", false
		}
	} else if json.Unmarshal(data, &top) != nil {
		return "", "", false
	}
	var bodyType string
	if raw := top["type"]; len(raw) > 0 && raw[0] == '"' {
		_ = json.Unmarshal(raw, &bodyType)
	}
	typ = bodyType
	if typ == "" && name != "message" {
		typ = name
	}
	sseName := name
	if sseName == "message" {
		sseName = ""
	}
	var b strings.Builder
	b.WriteString(strconv.Quote(sseName))
	for _, k := range []string{"type", "item_id", "index", "output_index", "content_index", "summary_index", "command_index"} {
		s := rawScalar(top[k])
		if s == "" || (k == "summary_index" || k == "command_index") && s == "n:0" {
			continue
		}
		b.WriteByte('\x00')
		b.WriteString(k)
		b.WriteByte('=')
		b.WriteString(strconv.Quote(s))
	}
	return typ, b.String(), true
}

var wholeResponsesEvents = map[string]bool{
	"response.created":                             true,
	"response.in_progress":                         true,
	"response.completed":                           true,
	"response.failed":                              true,
	"response.incomplete":                          true,
	"response.queued":                              true,
	"response.output_item.added":                   true,
	"response.output_item.done":                    true,
	"response.content_part.added":                  true,
	"response.content_part.done":                   true,
	"response.output_text.done":                    true,
	"response.output_text.annotation.added":        true,
	"response.refusal.done":                        true,
	"response.function_call_arguments.done":        true,
	"response.file_search_call.in_progress":        true,
	"response.file_search_call.searching":          true,
	"response.file_search_call.completed":          true,
	"response.web_search_call.in_progress":         true,
	"response.web_search_call.searching":           true,
	"response.web_search_call.completed":           true,
	"response.reasoning_summary_part.added":        true,
	"response.reasoning_summary_part.done":         true,
	"response.reasoning_summary_text.done":         true,
	"response.reasoning_text.done":                 true,
	"response.image_generation_call.in_progress":   true,
	"response.image_generation_call.generating":    true,
	"response.image_generation_call.partial_image": true,
	"response.image_generation_call.completed":     true,
	"response.mcp_call.in_progress":                true,
	"response.mcp_call.completed":                  true,
	"response.mcp_call.failed":                     true,
	"response.mcp_call_arguments.done":             true,
	"response.mcp_list_tools.in_progress":          true,
	"response.mcp_list_tools.completed":            true,
	"response.mcp_list_tools.failed":               true,
	"response.code_interpreter_call.in_progress":   true,
	"response.code_interpreter_call.interpreting":  true,
	"response.code_interpreter_call.completed":     true,
	"response.code_interpreter_call_code.done":     true,
	"response.custom_tool_call_input.done":         true,
	"response.audio.done":                          true,
	"response.audio.transcript.done":               true,
	"response.output_audio.done":                   true,
	"response.output_audio_transcript.done":        true,
	"response.shell_call_output_content.done":      true,
}

// wholeField says the string at steps sits in a known complete (non-streamed)
// protocol field so its trailing letters never carry onto the next event:
// Gemini finishReason ("STOP" with a "public-ip-" stand-in went out as "STO"
// and "PSTOP"), safetyRatings, groundingMetadata, promptFeedback,
// usageMetadata, Chat system_fingerprint/logprobs/message/annotations,
// Anthropic citations and lifecycle events, and non-delta or audio Responses
// events. Unknown events, proxy extensions, object keys shaped like array
// indices, empty parent keys and literal dotted keys keep streaming normally.
func wholeField(event string, steps []pathStep) bool {
	if len(steps) == 0 {
		return false
	}
	if strings.HasPrefix(event, "response.") {
		switch event {
		case "response.audio.delta", "response.output_audio.delta":
			if len(steps) == 1 && !steps[0].arr && steps[0].key == "delta" {
				return true
			}
		case "response.output_text.delta":
			if !steps[0].arr && steps[0].key == "logprobs" {
				return true
			}
		default:
			if wholeResponsesEvents[event] {
				return true
			}
		}
	}
	switch event {
	case "message_start":
		if !steps[0].arr && steps[0].key == "message" {
			return true
		}
	case "message_delta":
		if !steps[0].arr && (steps[0].key == "delta" || steps[0].key == "usage") {
			return true
		}
	case "content_block_start":
		if !steps[0].arr && steps[0].key == "content_block" {
			return true
		}
	case "content_block_delta":
		if len(steps) >= 3 && !steps[0].arr && steps[0].key == "delta" && !steps[1].arr && steps[1].key == "citation" {
			return true
		}
	}
	if len(steps) == 1 && !steps[0].arr && (steps[0].key == "system_fingerprint" || steps[0].key == "service_tier") {
		return true
	}
	if len(steps) >= 4 && !steps[0].arr && steps[0].key == "choices" && steps[1].arr && !steps[2].arr {
		switch steps[2].key {
		case "logprobs", "message":
			return true
		case "delta":
			if !steps[3].arr {
				if steps[3].key == "annotations" && len(steps) >= 5 {
					return true
				}
				if len(steps) == 6 && steps[3].key == "reasoning_details" && steps[4].arr && !steps[5].arr && steps[5].key == "format" {
					return true
				}
			}
		}
	}
	g := steps
	if len(g) > 1 && !g[0].arr && g[0].key == "response" {
		g = g[1:]
	}
	if len(g) == 1 && !g[0].arr {
		switch g[0].key {
		case "modelVersion", "model_version", "responseId", "response_id", "createTime", "create_time", "traceId", "trace_id":
			return true
		}
	}
	if len(g) >= 2 && !g[0].arr {
		switch g[0].key {
		case "promptFeedback", "prompt_feedback", "usageMetadata", "usage_metadata":
			return true
		}
	}
	if len(g) >= 3 && !g[0].arr && g[0].key == "candidates" && g[1].arr && !g[2].arr {
		if len(g) == 3 {
			switch g[2].key {
			case "finishReason", "finishMessage", "finish_message", "continuationToken", "continuation_token":
				return true
			}
		} else {
			switch g[2].key {
			case "safetyRatings", "safety_ratings",
				"groundingMetadata", "grounding_metadata",
				"citationMetadata", "citation_metadata",
				"logprobsResult", "logprobs_result",
				"urlContextMetadata", "url_context_metadata":
				return true
			}
		}
	}
	return false
}

type objMeta struct {
	index        string
	jsonPath     string
	id           string
	thought      bool
	willContinue bool
}

type pieceStr struct {
	key   string
	steps []pathStep
}

// textSeg is one typed segment of a string's stream identity, keeping object
// keys separate from array elements and protocol identifiers so no JSON key
// bytes can collide with internal markers.
type textSeg struct {
	kind    byte   // 'k' object key, 'a' array element
	key     string // object key (for 'k'), or array slot (for 'a')
	idKind  byte   // 0 none, '#' index, '@' jsonPath, '$' id
	id      string // explicit index/jsonPath/id
	thought bool   // Gemini part thought flag
}

func writeLenStr(b *strings.Builder, s string) {
	b.WriteString(strconv.Itoa(len(s)))
	b.WriteByte(':')
	b.WriteString(s)
}

// segKey encodes a path of typed segments into an injective map key so O(N)
// matching never collides on any JSON key bytes (dots, colons, control chars).
// When an array element carries an explicit protocol identifier (index,
// jsonPath, or id), its array slot is omitted so reordered or duplicate
// entries compare by identity rather than position.
func segKey(segs []textSeg) string {
	var b strings.Builder
	for _, s := range segs {
		b.WriteByte(s.kind)
		if s.kind != 'a' || s.idKind == 0 {
			writeLenStr(&b, s.key)
		} else {
			b.WriteString("0:")
		}
		b.WriteByte(s.idKind)
		writeLenStr(&b, s.id)
		if s.thought {
			b.WriteByte('1')
		} else {
			b.WriteByte('0')
		}
	}
	return b.String()
}

// pieces says which text of a stream each string of an event is a piece
// of. A string's text is its path, with each object in an array named by
// what the protocol names it by rather than where it stands: Chat
// Completions' tool call by its index (the next call's first piece sits
// where the last one's did, at tool_calls.0), Gemini's candidate by its
// index, a streamed argument by its jsonPath, and a Gemini part as a thought
// or not (the answer comes at parts.0.text after the thoughts).
// In a Gemini part only text is streamed in pieces, and a functionCall's
// partialArgs value while willContinue says more of it is to come: a
// functionCall's args, a functionResponse's response, code and its result
// come whole, each part a call or a result of its own, so nothing of one
// is held for the next (#7837: the "d" of one read_file's "/tmp/foo1.md"
// went to the start of the next one's Path).
type pieces struct {
	event string
	objs  map[int]*objMeta
	strs  []pieceStr // every string's key and exact path steps, in walk order
}

func normIndex(s string, bare bool) string {
	if bare {
		if s == "null" {
			return "b:null"
		}
		if len(s) > 0 && (s[0] == '-' || s[0] >= '0' && s[0] <= '9') {
			return "n:" + normNum(s)
		}
		return "b:" + s
	}
	if len(s) > 0 && (s[0] == '-' || s[0] >= '0' && s[0] <= '9') {
		if json.Valid([]byte(s)) {
			return "n:" + normNum(s)
		}
	}
	return "s:" + s
}

func altIDsConflict(prev, cur []string) bool {
	if len(cur) == 0 {
		return false
	}
	for i, cid := range cur {
		if cid == "" {
			continue
		}
		if i >= len(prev) || prev[i] != cid {
			return true
		}
	}
	return false
}

func piecesOf(data []byte, event string) *pieces {
	p := &pieces{event: event}
	meta := func(obj int) *objMeta {
		if p.objs == nil {
			p.objs = map[int]*objMeta{}
		}
		m := p.objs[obj]
		if m == nil {
			m = &objMeta{}
			p.objs[obj] = m
		}
		return m
	}
	var w walker
	w = walker{
		in:    data,
		track: true,
		bare: func(_, key, raw string) {
			n := len(w.steps)
			if n < 2 || w.steps[n-1].arr || w.steps[n-2].obj == 0 {
				return
			}
			obj := w.steps[n-2].obj
			switch key {
			case "index":
				meta(obj).index = normIndex(raw, true)
			case "id":
				meta(obj).id = "b:" + raw
			case "thought":
				meta(obj).thought = raw == "true"
			case "willContinue", "will_continue":
				meta(obj).willContinue = raw == "true"
			}
		},
		fn: func(_, key, s string) string {
			p.strs = append(p.strs, pieceStr{key: key, steps: slices.Clone(w.steps)})
			n := len(w.steps)
			if n < 2 || w.steps[n-1].arr || w.steps[n-2].obj == 0 {
				return s
			}
			obj := w.steps[n-2].obj
			switch key {
			case "index":
				meta(obj).index = normIndex(s, false)
			case "jsonPath", "json_path":
				meta(obj).jsonPath = "p:" + s
			case "id":
				meta(obj).id = "s:" + s
			}
			return s
		},
	}
	w.value(w.ws(0), "", "")
	return p
}

// of returns the canonical text key of the ord-th string in walk order, any
// secondary explicit ids along indexed array segments (so two calls sharing
// index:0 with different explicit ids never collide, while a follow-up chunk
// omitting id still matches), piece (continues a prior event), and more (the
// next event may continue it).
func (p *pieces) of(ord int) (key string, altIDs []string, piece, more bool) {
	if ord < 0 || ord >= len(p.strs) {
		return "", nil, false, false
	}
	steps := p.strs[ord].steps
	if wholeField(p.event, steps) {
		return "", nil, false, false
	}
	segs := make([]textSeg, len(steps))
	part := -1 // where a Gemini part's own keys start
	for i, st := range steps {
		isLeaf := i == len(steps)-1
		if st.arr {
			seg := textSeg{kind: 'a', key: st.key}
			if !isLeaf {
				if m := p.objs[st.obj]; m != nil {
					switch {
					case m.index != "":
						seg.idKind, seg.id = '#', m.index
						if m.index != "b:null" && m.id != "" && m.id != "s:" && m.id != "b:null" {
							if altIDs == nil {
								altIDs = make([]string, len(steps))
							}
							altIDs[i] = m.id
						}
					case m.jsonPath != "":
						seg.idKind, seg.id = '@', m.jsonPath
					case m.id != "":
						seg.idKind, seg.id = '$', m.id
					}
					seg.thought = m.thought
				}
				if part < 0 && i > 0 && !steps[i-1].arr && steps[i-1].key == "parts" {
					part = i + 1
				}
			}
			segs[i] = seg
		} else {
			seg := textSeg{kind: 'k', key: st.key}
			if !isLeaf {
				if m := p.objs[st.obj]; m != nil {
					if m.index != "" {
						seg.idKind, seg.id = '#', m.index
					}
					if m.id != "" && m.id != "s:" && m.id != "b:null" {
						if altIDs == nil {
							altIDs = make([]string, len(steps))
						}
						altIDs[i] = m.id
					}
				}
			}
			segs[i] = seg
		}
	}
	key = segKey(segs)
	if part < 0 {
		return key, altIDs, true, true
	}
	own := steps[part:]
	switch {
	case len(own) == 1 && !own[0].arr && own[0].key == "text":
		return key, altIDs, true, true
	case len(own) == 4 && !own[0].arr && (own[0].key == "functionCall" || own[0].key == "function_call") &&
		!own[1].arr && (own[1].key == "partialArgs" || own[1].key == "partial_args") && own[2].arr &&
		!own[3].arr && (own[3].key == "stringValue" || own[3].key == "string_value"):
		m := p.objs[own[2].obj]
		return key, altIDs, true, m != nil && m.willContinue
	}
	return key, nil, false, false
}

// on matches each held tail to the unique string in this event (by walk
// ordinal) that continues it in O(N). A text two strings here could go on with
// goes on with neither.
func (p *pieces) on(tails []heldTail) (map[int]string, []bool) {
	type hit struct {
		ord    int
		altIDs []string
	}
	exact := map[string]hit{}
	exactHits := map[string]int{}
	for i, s := range p.strs {
		if kept[s.key] || s.key == "obfuscation" {
			continue
		}
		key, altIDs, piece, _ := p.of(i)
		if !piece {
			continue
		}
		exact[key] = hit{ord: i, altIDs: altIDs}
		exactHits[key]++
	}
	var carry map[int]string
	var used []bool
	for ti, ht := range tails {
		if exactHits[ht.key] == 1 {
			h := exact[ht.key]
			if altIDsConflict(ht.altIDs, h.altIDs) {
				continue
			}
			if carry == nil {
				carry = map[int]string{}
				used = make([]bool, len(tails))
			}
			carry[h.ord] = ht.tail
			used[ti] = true
		}
	}
	return carry, used
}
