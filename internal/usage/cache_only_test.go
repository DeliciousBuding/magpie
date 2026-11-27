package usage

import (
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

// A call can spend only cache tokens, including a failed try that handed on.
// Each usage view prices those tokens or marks the missing model price.
func TestCacheOnlyCallPricing(t *testing.T) {
	holdsHome(t)
	now := time.Date(2026, 10, 8, 12, 0, 0, 0, time.UTC)
	reset := now.Add(time.Hour)
	price := Pricer()
	priceSession := sessionRowPricer(settings.Settings{})
	for _, tc := range []struct {
		name              string
		read, write, hour int
		cost              float64
	}{
		{"read", 1e6, 0, 0, 0.5},
		{"write", 0, 1e6, 4e5, 7.75}, // 600k at $6.25/M, 400k at $10/M
	} {
		for _, model := range []string{"claude-opus-4-7", "unknown-cache-model"} {
			t.Run(tc.name+"/"+model, func(t *testing.T) {
				r := Record{Time: now, Provider: "relay", ProviderAccount: "acct", Model: model, Agent: "magpie-search/1", Session: "s", Status: 502, HandedOn: true,
					CacheRead: tc.read, CacheWrite: tc.write, CacheWrite1h: tc.hour}
				known := model == "claude-opus-4-7"
				want, missing := tc.cost, 0
				if !known {
					want, missing = 0, 1
				}
				check := func(view string, cost float64, unpriced int) {
					t.Helper()
					if cost != want || unpriced != missing {
						t.Errorf("%s: cost %g, unpriced %d; want %g, %d", view, cost, unpriced, want, missing)
					}
				}
				checkRow := func(view string, row Row) {
					t.Helper()
					if row.Cost != want || row.Priced != known {
						t.Errorf("%s: cost %g, priced %t; want %g, %t", view, row.Cost, row.Priced, want, known)
					}
				}

				rows, sum, _, _ := ledgerWith(time.Time{}, Filter{}, []Record{r}, nil)
				checkRow("ledger row", rows[0])
				check("ledger totals", sum.Cost, sum.Unpriced)
				checkRow("session row", priceSession(r, "log"))
				var group Group
				group.addOwn(r, price(r))
				if group.Own.Cost != want || group.Own.Tokens != 0 || group.Own.Calls != 0 {
					t.Errorf("own share %+v, want cost %g with no extra call or non-cache tokens", group.Own, want)
				}

				var block rowChunk
				block.add(Row{Record: r}, "", 0, false)
				b := newBlockPricer(provider.Renames{}, nil, price)
				defer b.release()
				row := b.block(&block).row(0)
				checkRow("gateway block", row)
				var sumRows Totals
				sumRows.addRow(row)
				check("page totals", sumRows.Cost, sumRows.Unpriced)
				var heat heatDays
				heat.add(&row, time.Time{}, time.UTC)
				d := heat.list()[0]
				check("heatmap", d.Cost, d.Unpriced)

				ss, days := buildGatewaySessions(&logSnapshot{blocks: []*rowChunk{&block}}, time.Time{}, nil, nil)
				check("gateway session", ss[0].Cost, ss[0].Unpriced)
				m := ss[0].Models[0]
				if m.Cost != want || m.Priced != known || days[0].Cost != want || days[0].Priced != known {
					t.Errorf("gateway model/day: %g/%t, %g/%t; want %g/%t", m.Cost, m.Priced, days[0].Cost, days[0].Priced, want, known)
				}

				qs := []provider.SubscriptionQuota{{Provider: r.Provider, User: r.ProviderAccount, Windows: []provider.QuotaWindow{{Name: "5 hours", Span: 5 * time.Hour, ResetsAt: &reset, Used: 50}}}}
				w := windowHolds(qs, now, now.Add(-24*time.Hour), price, provider.Renames{}, nil, nil, func(add func(Record)) { add(r) })[0].Windows[0]
				if h := w.Holds; h == nil {
					t.Errorf("no window holds: %s", w.HoldsWhy)
				} else if h.Priced != known || h.Routed.Cost != want || h.Cost != 2*want || h.Routed.CacheRead != int64(tc.read) || h.Routed.CacheWrite != int64(tc.write) {
					t.Errorf("window holds %+v, want priced %t, routed cost %g, window cost %g", h, known, want, 2*want)
				}
			})
		}
	}
}

func TestCacheOnlyReportedCost(t *testing.T) {
	holdsHome(t)
	var raw rowChunk
	r := Record{Time: time.Now(), Source: SourceReported, CacheRead: 200, OwnCost: new(0.75)}
	raw.add(Row{Record: r, Source: r.Source}, "", 0, false)
	b := newBlockPricer(provider.Renames{}, nil, Pricer())
	defer b.release()
	if row := b.block(&raw).row(0); !row.Priced || row.Cost != *r.OwnCost {
		t.Errorf("reported cache read: cost %g, priced %t; want 0.75, true", row.Cost, row.Priced)
	}
}
