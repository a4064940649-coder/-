# -*- coding: utf-8 -*-
"""Freeze the event table BEFORE any return is computed. Writes results.json['EVENTS_FROZEN']."""
import os, json, hashlib, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results.json')
res = json.load(open(R)) if os.path.exists(R) else {}
assert 'EVENTS_FROZEN' not in res, 'already frozen'
sha = lambda f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
summ = json.load(open(os.path.join(HERE, 'events_build_summary.json')))
nx = json.load(open(os.path.join(HERE, 'news_xcheck.json')))

# Manual review of every news item the automatic matcher could not find (news_xcheck_*.csv, covered windows).
manual = {
    'SEO': {
        'not_issuer_event (participates in another firm SEO / M&A / capex)': ['1301 2019-06-11', '1314 2020-09-14', '1439 2020-09-29', '1605 2018-02-23',
                                                                           '2049 2015-04-30', '2474 2018-03-15', '3481 2015-05-05', '8150 2014-08-13'],
        'name->code mapping error in news parser': ['2020 2020-07-17 (三商壽=2867)', '2317 2018-05-23 (大同=2371)'],
        'GDR / shareholder authorisation, not a domestic SEO filing': ['2327 2020-03-02', '2409 2014-03-12', '2448 2015-06-29'],
        'filing falls in FSC mirror gap (2013-11-22~12-31)': ['3561 2014-01-21'],
        'plausible genuine miss': ['3474 2014-02-14 華亞科', '2841 2018-09-26 台開 (may never have been filed)', '3711 2020-03-30 日月光投控 (may never have been filed)'],
    },
    'CB': {
        'ECB / overseas, not on TPEx domestic CB list (by design)': ['2311 2013-09-25', '2327 2020-03-02', '2883 2019-01-21 (台郡 ECB)', '3481 2019-12-17', '3665 2019-10-01',
                                                                     '4958 2020-06-04', '6269 2018-12-19', '6269 2020-12-25', '8039 2021-07-29', '8039 2021-11-25'],
        'false positive (market commentary / ECB=European Central Bank / other firm)': ['2855 2016-12-12', '8101 2016-12-09', '2017 2016-08-26', '6641 2021-10-05', '5469 2020-03-30'],
        'withdrawn/cancelled per FSC (廢止/撤銷) - correctly absent': ['2348 2020-06-19', '2542 2015-09-01', '6655 2019-06-30', '6655 2019-09-12'],
        'name->code mapping error': ['3308 2018-07-13 (聯德-KY=4912)', '3308 2020-05-20', '2020 2021-04-08 (皇龍)', '2317 2019-12-30 (鴻海 straight bond)'],
        'no filing found (likely never issued)': ['3711 2020-03-30'],
    },
}
seo_true = (nx['SEO']['n_in_covered_window'] - 17) + 3   # found + plausible genuine misses
seo_cov_true = (nx['SEO']['n_in_covered_window'] - 17) / seo_true
cb_true_found = nx['CB']['n_in_covered_window'] - 24
cb_cov_true = cb_true_found / (cb_true_found + 0)

res['EVENTS_FROZEN'] = dict(
    time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    note='Frozen before any stock return was computed (build_fsc.py, build_events.py, news_xcheck.py read only filings, CB quotes and news text).',
    sha256={f: sha(f) for f in ['events.csv', 'fsc_cases_all.csv', 'tpex_cb_listings.csv', 'news_events.csv']},
    tier='第 1 層（官方申報彙總表 + 櫃買 CB 日行情封存，皆為 GitHub 鏡像）',
    n_by_type=summ['by_type'],
    coverage=dict(
        fsc_cb_to_tpex_listing=summ['coverage_fsc_cb_to_tpex_listing'], n_fsc_cb=summ['n_fsc_cb_2007_2025'],
        tpex_listing_to_fsc=summ['coverage_tpex_listing_to_fsc_cb'], n_tpex=summ['n_tpex_in_fsc_window'],
        news_sample_SEO_auto=nx['SEO']['coverage'], news_sample_SEO_true_after_manual=round(seo_cov_true, 3), news_SEO_true_n=seo_true,
        news_sample_CB_auto=nx['CB']['coverage'], news_sample_CB_true_after_manual=round(cb_cov_true, 3), news_CB_true_n=cb_true_found),
    gaps=dict(SEO_SDIV_missing=['2009-08-25~2009-12-31', '2010-07-30~2010-12-31', '2013-11-22~2013-12-31', '2016-01-01~2016-12-31', '2025-07-26~2026-10-27'],
              CB='TPEx listings complete 2007-01-03~2026-10-27; FSC filing/eff dates missing in the same gaps (listing date still known)'),
    manual_review=manual)
json.dump(res, open(R, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res['EVENTS_FROZEN'], ensure_ascii=False, indent=1)[:2500])
