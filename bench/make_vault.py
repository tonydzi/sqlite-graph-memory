# -*- coding: utf-8 -*-
"""make_vault.py — generate the synthetic mini-vault and its question set.

Everything in bench/vault/ and bench/questions.jsonl comes out of this file. There is no
personal data in it: a fictional lab with invented people, teams, sites, tools and
projects, written as markdown notes that link to each other with [[wikilinks]] the way a
hand-kept vault does. The output is committed, so nobody needs to run this to use the
bench; it exists so the vault is reviewable as code and CI can check it was not edited by
hand (`--check`).

Deterministic: same seed, same bytes. Change the generator, and the vault hash changes,
and every result file on the leaderboard becomes stale on purpose (see validate.py).

Question classes (the `cls` field):

    fact              one note holds the answer
    bridge            the answer needs two notes joined by a wikilink
    paraphrase        names no entity: describes a project's goal in other words.
                      Keyword search is meant to fail here
    paraphrase_bridge the same description, then one wikilink hop to the tool's language
    temporal          a decision was later replaced; the question asks for the current
                      or the original value, and both notes look alike
    absent_entity     the person / project / tool / site does not exist in the vault
    absent_attribute  the entity exists, the asked-for attribute is never written down

The two `absent_*` classes have no answer. A system that answers them anyway is
confabulating, and that is scored (see bench_core.score).

USAGE
    python bench/make_vault.py            # (re)write bench/vault and bench/questions.jsonl
    python bench/make_vault.py --check    # exit 1 if the committed files differ from output
"""
import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

SEED = 20261009
HERE = Path(__file__).resolve().parent

FIRST = ['Ilsa', 'Tomas', 'Mirela', 'Oskar', 'Petra', 'Anselm', 'Linnea', 'Corvin', 'Yara',
         'Bastian', 'Elodie', 'Rurik', 'Saskia', 'Teodor', 'Joran', 'Ottilie', 'Kasimir',
         'Nadia', 'Leopold', 'Ines', 'Fabian', 'Greta', 'Hollis', 'Marit', 'Severin', 'Talia',
         'Ulrik', 'Vesna', 'Emeric', 'Brisa']
LAST = ['Moravec', 'Quintanar', 'Halloran', 'Vestergaard', 'Okafor-Lind', 'Brannock',
        'Lindqvist', 'Asturias', 'Fennimore', 'Kalvaitis', 'Thorsby', 'Delacourt', 'Marchetti',
        'Oyelaran', 'Strandberg', 'Voskuijl', 'Achterberg', 'Pellegrin', 'Tamberlane',
        'Wexley', 'Ibarrondo', 'Szollosi', 'Rautio', 'Carrow']
SITES = ['Port Varen', 'Holloway Reach', 'Ketterby', 'Amsel Quay', 'Drumlin Cross',
         'Sorrel Point', 'Larkspur Basin', 'Owlsgate', 'Tamsin Ford', 'Brackwater',
         'Ninefold', 'Calder Rise']
ABSENT_SITES = ['Wexmoor', 'Gullhaven', 'Sallow Mere']
REGIONS = ['northern coast', 'river delta', 'high plateau', 'eastern valleys', 'lake district']
TEAMS = ['Lighthouse', 'Tidewater', 'Bramble', 'Ironbark', 'Saltmarsh', 'Foxglove', 'Granite',
         'Meridian', 'Cinder', 'Willowherb']
TOOLS = ['Quillmark', 'Ferrodb', 'Lanternfish', 'Corvex', 'Tallyhoe', 'Siltcore', 'Brindlewood',
         'Mosscache', 'Pennantry', 'Gristmill', 'Wickline', 'Harrowby', 'Spindrift',
         'Cobbleset', 'Thimblerig', 'Orreryx', 'Sextantix', 'Loamstack', 'Kilnworth', 'Fathomer']
ABSENT_TOOLS = ['Marrowgate', 'Pikeline', 'Dovetailor']
LANGS = ['Rust', 'OCaml', 'Elixir', 'Haskell', 'Zig', 'Kotlin', 'Erlang', 'Clojure', 'Scala',
         'Fortran']
PROJECTS = ['Kestrel', 'Heron', 'Plover', 'Merlin', 'Osprey', 'Curlew', 'Dunlin', 'Gannet',
            'Shrike', 'Bittern', 'Godwit', 'Lapwing', 'Avocet', 'Tern', 'Skua', 'Petrel',
            'Dipper', 'Siskin', 'Linnet', 'Ouzel', 'Whimbrel', 'Redshank', 'Garganey', 'Smew',
            'Nightjar']
ABSENT_PROJECTS = ['Albatross', 'Cormorant', 'Pipit']
GOALS = [
    ('cut cold-start latency for the search cluster', 'cold-start latency'),
    ('replace nightly batch exports with a streaming feed', 'streaming feed'),
    ('move invoice reconciliation off spreadsheets', 'invoice reconciliation'),
    ('give field staff offline access to site maps', 'offline access to site maps'),
    ('detect sensor drift in the greenhouse arrays', 'sensor drift'),
    ('shrink the backup window below one hour', 'backup window'),
    ('translate support tickets between five languages', 'support tickets'),
    ('automate desk booking across all sites', 'desk booking'),
    ('forecast spare-part demand for the workshop', 'spare-part demand'),
    ('archive twenty years of lab notebooks', 'lab notebooks'),
    ('catch duplicate supplier records before payment', 'duplicate supplier records'),
    ('route night-shift alerts to the right on-call person', 'night-shift alerts'),
    ('measure energy use per experiment', 'energy use per experiment'),
    ('replace paper permits for the loading dock', 'paper permits'),
    ('index every recorded seminar by speaker and topic', 'recorded seminar'),
    ('predict which builds will fail before they run', 'builds will fail'),
    ('keep the visitor badge system running during outages', 'visitor badge system'),
    ('compress microscope images without losing detail', 'microscope images'),
    ('track loaned equipment between sites', 'loaned equipment'),
    ('summarize weekly safety inspections', 'safety inspections'),
    ('unify the three internal wikis into one', 'three internal wikis'),
    ('schedule shared GPU time fairly', 'shared GPU time'),
    ('flag expiring chemical stock', 'expiring chemical stock'),
    ('replace the fax line to the customs broker', 'fax line'),
    ('measure how long onboarding really takes', 'onboarding'),
]
# The same goals said with (almost) no shared content words. Questions built from these
# cannot be answered by matching the note's vocabulary: keyword search should fail them,
# and that is what they are for.
PARAPHRASE = {
    'cut cold-start latency for the search cluster': 'make the first lookups of the morning come back faster',
    'replace nightly batch exports with a streaming feed': 'stop waiting overnight for data dumps and push changes the moment they happen',
    'move invoice reconciliation off spreadsheets': 'get accountants out of Excel when matching bills to payments',
    'give field staff offline access to site maps': 'let people working somewhere without signal still see where things are',
    'detect sensor drift in the greenhouse arrays': 'notice when the plant-house instruments slowly start lying',
    'shrink the backup window below one hour': 'make copying everything for safekeeping finish in under sixty minutes',
    'translate support tickets between five languages': 'let customers write help requests in their own tongue',
    'automate desk booking across all sites': 'stop colleagues fighting over where to sit',
    'forecast spare-part demand for the workshop': 'guess which replacement components the repair shop will need',
    'archive twenty years of lab notebooks': 'preserve two decades of handwritten research records',
    'catch duplicate supplier records before payment': 'avoid paying one vendor twice because it is listed twice',
    'route night-shift alerts to the right on-call person': 'wake the correct engineer when something breaks at 3 a.m.',
    'measure energy use per experiment': 'know how much electricity each trial consumes',
    'replace paper permits for the loading dock': 'digitize the forms truck drivers sign at the delivery bay',
    'index every recorded seminar by speaker and topic': 'make old talk videos findable by who spoke and about what',
    'predict which builds will fail before they run': 'know in advance that a compilation is going to break',
    'keep the visitor badge system running during outages': 'let guests into the building even while the network is down',
    'compress microscope images without losing detail': 'make huge magnified pictures smaller while keeping them sharp',
    'track loaned equipment between sites': 'know which borrowed gear sits at which location',
    'summarize weekly safety inspections': 'condense the hazard walkthroughs done every seven days',
    'unify the three internal wikis into one': 'merge the competing documentation sites',
    'schedule shared GPU time fairly': 'split the graphics-card cluster between groups without favouritism',
    'flag expiring chemical stock': 'warn before reagents on the shelf go out of date',
    'replace the fax line to the customs broker': 'retire the old paper-transmission link with the import agent',
    'measure how long onboarding really takes': 'find out how many days new hires need before they are productive',
}
ROLES = ['data engineer', 'research scientist', 'site reliability engineer', 'product designer',
         'technical writer', 'ML engineer', 'security analyst', 'program manager',
         'hardware technician', 'statistician']
HOBBIES = [
    ('restores old sailing dinghies', 'sailing dinghies'),
    ('keeps bees on the roof of the east building', 'bees'),
    ('plays the cello in an amateur quartet', 'cello'),
    ('runs a small seed library for heirloom tomatoes', 'seed library'),
    ('builds mechanical clocks from kits', 'mechanical clocks'),
    ('coaches a junior fencing club', 'fencing'),
    ('photographs abandoned railway stations', 'railway stations'),
    ('brews cider from foraged apples', 'cider'),
    ('swims in the harbour every morning, all year', 'swims in the harbour'),
    ('binds books by hand', 'binds books'),
    ('maps lichens on old stone walls', 'lichens'),
    ('races homing pigeons', 'homing pigeons'),
    ('restores pinball machines', 'pinball machines'),
    ('teaches evening classes in calligraphy', 'calligraphy'),
    ('collects antique slide rules', 'slide rules'),
    ('sings in a sea-shanty choir', 'sea-shanty choir'),
]
CONCEPTS = [
    ('graceful degradation', 'A system keeps doing the most important part of its job when a dependency fails, instead of failing as a whole.'),
    ('backpressure', 'A slow consumer signals upstream to send less, so queues do not grow until something falls over.'),
    ('idempotent writes', 'Repeating the same write leaves the data in the same state, which makes retries safe.'),
    ('cold-start caching', 'Warm the cache before traffic arrives so the first users of the day do not pay for an empty cache.'),
    ('feature flags', 'Ship code switched off and turn it on for a few users first, so a bad change can be undone without a deploy.'),
    ('schema evolution', 'Change the shape of stored data in steps that old and new readers both understand.'),
    ('rate limiting', 'Cap how often one caller may ask for work, so one noisy client cannot starve the rest.'),
    ('observability budget', 'Decide up front how much logging and tracing a service may cost, and spend it where incidents happen.'),
    ('chaos drills', 'Break things on purpose, on a schedule, while people are watching, to learn how the system really fails.'),
    ('data lineage', 'Record where every derived number came from, so a wrong figure can be traced back to its source.'),
    ('shadow traffic', 'Copy real requests to a new version that answers into the void, and compare it with the old one.'),
    ('blameless review', 'After an incident, ask what made the mistake easy to make, not who made it.'),
]
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
FILLER = [
    '{first} prefers written proposals to meetings and usually answers them within a day.',
    'Colleagues describe {first} as patient with new joiners.',
    '{first} keeps a running list of open questions at the top of every working document.',
    'Most of what {first} ships starts as a one-page note.',
    '{first} has asked twice for a quieter desk.',
    'When something breaks, {first} writes the timeline first and the fix second.',
]


def slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


def link(nid, label):
    return '[[%s|%s]]' % (nid, label)


def plain(s):
    """Wikilink markup -> its visible text. Same rule as bench_core.normalize."""
    return re.sub(r'\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]', lambda m: m.group(2) or m.group(1), s)


def fmt_date(y, m, d):
    return '%d %s %d' % (d, MONTHS[m - 1], y)


def note(title, typ, date, body, extra=None):
    fm = ['---', 'title: "%s"' % title, 'type: %s' % typ, 'date: %s' % date, 'synthetic: true']
    for k, v in (extra or {}).items():
        fm.append('%s: %s' % (k, v))
    fm.append('---')
    return '\n'.join(fm) + '\n\n# %s\n\n' % title + body.strip() + '\n'


def build(seed=SEED):
    R = random.Random(seed)
    notes = {}        # id -> text
    qs = []           # question dicts

    def q(cls, question, answers, support, evidence=None):
        qs.append({'cls': cls, 'question': question, 'answers': answers,
                   'support': support, 'evidence': evidence or []})

    # ------------------------------------------------------------- names
    pairs = [(f, l) for f in FIRST for l in LAST]
    R.shuffle(pairs)
    used_first, people_names = {}, []
    for f, l in pairs:
        if used_first.get(f, 0) >= 2:      # some shared first names = realistic ambiguity
            continue
        used_first[f] = used_first.get(f, 0) + 1
        people_names.append('%s %s' % (f, l))
        if len(people_names) == 50 + 4:
            break
    absent_people = people_names[50:]
    people_names = people_names[:50]

    sites = [{'name': s, 'id': 'site-' + slug(s), 'region': R.choice(REGIONS),
              'opened': R.randint(1998, 2021), 'desks': R.randint(18, 240)} for s in SITES]
    tools = [{'name': t, 'id': 'tool-' + slug(t), 'lang': LANGS[i % len(LANGS)],
              'version': '%d.%d.%d' % (R.randint(1, 9), R.randint(0, 12), R.randint(0, 9)),
              'first': R.randint(2009, 2023)} for i, t in enumerate(TOOLS)]
    R.shuffle(tools)
    teams = [{'name': t, 'id': 'team-' + slug(t), 'site': sites[i % len(sites)],
              'formed': R.randint(2005, 2022), 'tools': tools[2 * i:2 * i + 2], 'members': []}
             for i, t in enumerate(TEAMS)]
    for t in teams:
        for tl in t['tools']:
            tl['team'] = t
    hobbies = HOBBIES[:]
    people = []
    for i, n in enumerate(people_names):
        team = teams[i % len(teams)]
        p = {'name': n, 'first': n.split()[0], 'id': 'person-' + slug(n), 'team': team,
             'role': R.choice(ROLES), 'joined': R.randint(2006, 2025), 'home': R.choice(sites),
             'tool': R.choice(tools), 'hobby': hobbies[i] if i < len(hobbies) else None}
        team['members'].append(p)
        people.append(p)
    for i, p in enumerate(people):
        p['mentor'] = people[(i * 7 + 3) % len(people)] if i % 3 else None
    for t in teams:
        t['head'] = t['members'][0]

    goals = GOALS[:]
    R.shuffle(goals)
    projects = []
    for i, name in enumerate(PROJECTS):
        projects.append({'name': name, 'title': 'Project ' + name, 'id': 'project-' + slug(name),
                         'lead': people[(i * 3 + 1) % len(people)], 'tool': tools[i % len(tools)],
                         'start': (R.randint(2019, 2025), R.randint(1, 12)), 'goal': goals[i],
                         'dep': None, 'concepts': []})
    for i, p in enumerate(projects):
        if i % 2 == 1:
            p['dep'] = projects[(i + 5) % len(projects)]
    concepts = [{'name': c, 'id': 'concept-' + slug(c), 'text': t, 'projects': []}
                for c, t in CONCEPTS]
    for i, p in enumerate(projects):
        for c in (concepts[i % len(concepts)], concepts[(i * 5 + 2) % len(concepts)]):
            if c not in p['concepts']:
                p['concepts'].append(c); c['projects'].append(p)

    # ------------------------------------------------------------- notes: sites
    for s in sites:
        ev_desks = 'The %s office has %d desks.' % (s['name'], s['desks'])
        s['ev_desks'] = ev_desks
        body = ('%s lies in the %s region. The %s office opened in %d. %s\n\n'
                'Teams based here: %s.\n' % (
                    s['name'], s['region'], s['name'], s['opened'], ev_desks,
                    ', '.join(link(t['id'], t['name']) for t in teams if t['site'] is s) or 'none at the moment'))
        notes[s['id']] = note(s['name'], 'site', '%d-01-01' % s['opened'], body)

    # ------------------------------------------------------------- notes: teams
    for t in teams:
        t['ev_site'] = 'The %s team is based in %s.' % (t['name'], link(t['site']['id'], t['site']['name']))
        t['ev_head'] = 'The %s team is headed by %s.' % (t['name'], link(t['head']['id'], t['head']['name']))
        body = ('%s %s The team was formed in %d.\n\n'
                'The %s team maintains %s.\n\n'
                'Members: %s.\n' % (
                    t['ev_site'], t['ev_head'], t['formed'], t['name'],
                    ' and '.join(link(x['id'], x['name']) for x in t['tools']),
                    ', '.join(link(m['id'], m['name']) for m in t['members'])))
        notes[t['id']] = note(t['name'] + ' team', 'team', '%d-01-01' % t['formed'], body)

    # ------------------------------------------------------------- notes: tools
    for tl in tools:
        tl['ev_lang'] = '%s is written in %s.' % (tl['name'], tl['lang'])
        tl['ev_ver'] = 'The current release of %s is version %s.' % (tl['name'], tl['version'])
        tl['ev_first'] = '%s was first released in %d.' % (tl['name'], tl['first'])
        tl['ev_team'] = '%s is maintained by the %s team.' % (tl['name'], link(tl['team']['id'], tl['team']['name']))
        users = [p for p in projects if p['tool'] is tl]
        body = ('%s %s\n\n%s %s\n' % (tl['ev_lang'], tl['ev_team'], tl['ev_first'], tl['ev_ver']))
        if users:
            body += '\nUsed by %s.\n' % ', '.join(link(p['id'], p['title']) for p in users)
        notes[tl['id']] = note(tl['name'], 'tool', '%d-01-01' % tl['first'], body)

    # ------------------------------------------------------------- notes: people
    for p in people:
        n = p['name']
        p['ev_join'] = '%s joined the lab in %d.' % (n, p['joined'])
        p['ev_home'] = '%s grew up in %s.' % (n, link(p['home']['id'], p['home']['name']))
        p['ev_tool'] = "%s's preferred tool is %s." % (n, link(p['tool']['id'], p['tool']['name']))
        p['ev_team'] = '%s works as a %s on the %s team.' % (n, p['role'], link(p['team']['id'], p['team']['name']))
        lines = [p['ev_team'], p['ev_join'], p['ev_home'], p['ev_tool']]
        if p['mentor']:
            lines.append('%s was mentored by %s.' % (n, link(p['mentor']['id'], p['mentor']['name'])))
        if p['hobby']:
            p['ev_hobby'] = 'Outside work, %s %s.' % (n, p['hobby'][0])
            lines.append(p['ev_hobby'])
        lines.append(R.choice(FILLER).format(first=p['first']))
        led = [x for x in projects if x['lead'] is p]
        if led:
            lines.append('%s currently leads %s.' % (p['first'], ', '.join(link(x['id'], x['title']) for x in led)))
        body = ' '.join(lines[:3]) + '\n\n' + ' '.join(lines[3:]) + '\n'
        notes[p['id']] = note(n, 'person', '%d-01-01' % p['joined'], body)

    # ------------------------------------------------------------- notes: projects
    for p in projects:
        y, m = p['start']
        p['ev_lead'] = '%s is led by %s.' % (p['title'], link(p['lead']['id'], p['lead']['name']))
        p['ev_start'] = '%s started in %s %d.' % (p['title'], MONTHS[m - 1], y)
        p['ev_tool'] = '%s is built on %s.' % (p['title'], link(p['tool']['id'], p['tool']['name']))
        p['ev_goal'] = 'The goal of %s is to %s.' % (p['title'], p['goal'][0])
        body = '%s %s %s\n\n%s' % (p['ev_goal'], p['ev_lead'], p['ev_start'], p['ev_tool'])
        if p['dep']:
            body += ' %s depends on %s.' % (p['title'], link(p['dep']['id'], p['dep']['title']))
        body += '\n\nRelated ideas: %s.\n' % ', '.join(link(c['id'], c['name']) for c in p['concepts'])
        notes[p['id']] = note(p['title'], 'project', '%d-%02d-01' % (y, m), body)

    # ------------------------------------------------------------- notes: concepts
    for c in concepts:
        body = '%s\n\nProjects that rely on it: %s.\n' % (
            c['text'], ', '.join(link(p['id'], p['title']) for p in c['projects']))
        notes[c['id']] = note(c['name'].capitalize(), 'concept', '2024-06-01', body)

    # ------------------------------------------------------------- notes: decisions
    temporal = projects[:10]
    for i, p in enumerate(temporal):
        y1, m1, d1 = 2025, R.randint(1, 6), R.randint(1, 28)
        y2, m2, d2 = 2025, R.randint(7, 12), R.randint(1, 28)
        dl1 = fmt_date(2026, R.randint(1, 6), R.randint(1, 28))
        dl2 = fmt_date(2026 + (i % 2), R.randint(7, 12), R.randint(1, 28))
        id1 = 'decision-%d-%02d-%02d-%s-deadline' % (y1, m1, d1, slug(p['name']))
        id2 = 'decision-%d-%02d-%02d-%s-deadline' % (y2, m2, d2, slug(p['name']))
        att = R.sample(people, 3)
        p['dl'] = (dl1, dl2, id1, id2)
        p['ev_dl1'] = 'On %s, the steering group set the deadline for %s to %s.' % (
            fmt_date(y1, m1, d1), link(p['id'], p['title']), dl1)
        p['ev_dl2'] = 'On %s, the steering group moved the deadline for %s to %s, replacing %s.' % (
            fmt_date(y2, m2, d2), link(p['id'], p['title']), dl2, link(id1, 'the earlier decision'))
        notes[id1] = note('%s deadline (%d-%02d-%02d)' % (p['title'], y1, m1, d1), 'decision',
                          '%d-%02d-%02d' % (y1, m1, d1),
                          p['ev_dl1'] + '\n\nPresent: %s.\n' % ', '.join(link(a['id'], a['name']) for a in att),
                          {'status': 'superseded', 'superseded_by': '"[[%s]]"' % id2})
        notes[id2] = note('%s deadline (%d-%02d-%02d)' % (p['title'], y2, m2, d2), 'decision',
                          '%d-%02d-%02d' % (y2, m2, d2),
                          p['ev_dl2'] + '\n\nPresent: %s.\n' % ', '.join(link(a['id'], a['name']) for a in att),
                          {'status': 'current'})
    budgeted = projects[10:22]
    for p in budgeted:
        y, m, d = 2025, R.randint(1, 12), R.randint(1, 28)
        p['credits'] = R.randint(4, 95) * 1000
        nid = 'decision-%d-%02d-%02d-%s-budget' % (y, m, d, slug(p['name']))
        p['budget_id'] = nid
        p['ev_budget'] = 'On %s, the steering group approved a budget of %s compute credits for %s.' % (
            fmt_date(y, m, d), '{:,}'.format(p['credits']), link(p['id'], p['title']))
        notes[nid] = note('%s budget' % p['title'], 'decision', '%d-%02d-%02d' % (y, m, d),
                          p['ev_budget'] + '\n')

    # ------------------------------------------------------------- notes: meetings (distractors)
    talk = [
        '- Discussed whether {tool} can take the {proj} load next quarter.',
        '- {name} raised concerns about the {proj} timeline.',
        '- Agreed to revisit {concept} next week.',
        '- {name} will write up the open questions on {tool}.',
        '- Short demo of {proj}; questions about cost, no decisions.',
        '- Reminder: desk moves at {site} start Monday.',
    ]
    for k in range(40):
        y, m, d = 2025, R.randint(1, 12), R.randint(1, 28)
        team = teams[k % len(teams)]
        att = R.sample(team['members'], min(3, len(team['members'])))
        nid = 'meeting-%d-%02d-%02d-%s-sync' % (y, m, d, slug(team['name']))
        if nid in notes:
            nid += '-%d' % k
        lines = []
        for t in R.sample(talk, 3):
            pr, tl, c = R.choice(projects), R.choice(team['tools']), R.choice(concepts)
            lines.append(t.format(tool=link(tl['id'], tl['name']), proj=link(pr['id'], pr['title']),
                                  name=R.choice(att)['name'], concept=link(c['id'], c['name']),
                                  site=link(team['site']['id'], team['site']['name'])))
        if k < len(temporal):
            p = temporal[k]
            lines.append('- Planning still assumes the %s date of %s.' % (link(p['id'], p['title']), p['dl'][0]))
        body = 'Attendees: %s.\n\n%s\n' % (', '.join(link(a['id'], a['name']) for a in att), '\n'.join(lines))
        notes[nid] = note('%s sync %d-%02d-%02d' % (team['name'], y, m, d), 'meeting',
                          '%d-%02d-%02d' % (y, m, d), body)

    # ------------------------------------------------------------- home note
    hubs = [('Teams', teams, lambda t: link(t['id'], t['name'])),
            ('Sites', sites, lambda s: link(s['id'], s['name'])),
            ('Projects', projects, lambda p: link(p['id'], p['title'])),
            ('Ideas', concepts, lambda c: link(c['id'], c['name']))]
    body = 'Entry point of the Larkfield Lab vault. Every person, tool and meeting is reachable from these.\n'
    for h, xs, f in hubs:
        body += '\n## %s\n\n%s\n' % (h, ', '.join(f(x) for x in xs))
    notes['home'] = note('Larkfield Lab', 'index', '2026-01-01', body)

    # ------------------------------------------------------------- questions
    for p in R.sample([p for p in people if p['hobby']], 8):
        kind = R.choice(['join', 'home', 'tool', 'hobby'])
        if kind == 'join':
            q('fact', 'In what year did %s start working at the lab?' % p['name'], [str(p['joined'])], [p['id']], [p['ev_join']])
        elif kind == 'home':
            q('fact', 'Where did %s grow up?' % p['name'], [p['home']['name']], [p['id']], [p['ev_home']])
        elif kind == 'tool':
            q('fact', 'Which tool does %s like to use most?' % p['name'], [p['tool']['name']], [p['id']], [p['ev_tool']])
        else:
            q('fact', 'What does %s do in their free time?' % p['name'], [p['hobby'][1]], [p['id']], [p['ev_hobby']])
    for tl in R.sample(tools, 6):
        kind = R.choice(['lang', 'ver', 'first'])
        if kind == 'lang':
            q('fact', 'Which programming language is %s implemented in?' % tl['name'], [tl['lang']], [tl['id']], [tl['ev_lang']])
        elif kind == 'ver':
            q('fact', 'What is the latest version of %s?' % tl['name'], [tl['version']], [tl['id']], [tl['ev_ver']])
        else:
            q('fact', 'When did %s come out for the first time?' % tl['name'], [str(tl['first'])], [tl['id']], [tl['ev_first']])
    for p in R.sample(projects, 6):
        kind = R.choice(['lead', 'start', 'goal'])
        if kind == 'lead':
            q('fact', 'Who is in charge of %s?' % p['title'], [p['lead']['name']], [p['id']], [p['ev_lead']])
        elif kind == 'start':
            q('fact', 'When did work on %s begin?' % p['title'], ['%s %d' % (MONTHS[p['start'][1] - 1], p['start'][0])], [p['id']], [p['ev_start']])
        else:
            q('fact', 'What problem is %s meant to solve?' % p['title'], [p['goal'][1]], [p['id']], [p['ev_goal']])
    for s in R.sample(sites, 4):
        q('fact', 'How many desks are there at the %s office?' % s['name'], [str(s['desks'])], [s['id']], [s['ev_desks']])
    for p in R.sample(budgeted, 4):
        q('fact', 'How many compute credits did %s get approved?' % p['title'],
          ['{:,}'.format(p['credits']), str(p['credits'])], [p['budget_id']], [p['ev_budget']])

    for p in R.sample([x for x in people if x['team']['head'] is not x], 5):   # else the answer is in the question
        q('bridge', 'Who is the head of the team %s works on?' % p['name'], [p['team']['head']['name']],
          [p['id'], p['team']['id']], [p['ev_team'], p['team']['ev_head']])
    for tl in R.sample(tools, 5):
        q('bridge', 'Where is the team that maintains %s located?' % tl['name'], [tl['team']['site']['name']],
          [tl['id'], tl['team']['id']], [tl['ev_team'], tl['team']['ev_site']])
    for p in R.sample(projects, 5):
        q('bridge', 'In which programming language is the tool behind %s written?' % p['title'], [p['tool']['lang']],
          [p['id'], p['tool']['id']], [p['ev_tool'], p['tool']['ev_lang']])

    for p in R.sample(projects, 10):
        q('paraphrase', 'Which project exists to %s?' % PARAPHRASE[p['goal'][0]], [p['title']], [p['id']], [p['ev_goal']])
    for p in R.sample(projects, 6):
        q('paraphrase_bridge', 'In which language is the tool written that powers the project meant to %s?'
          % PARAPHRASE[p['goal'][0]], [p['tool']['lang']], [p['id'], p['tool']['id']], [p['ev_goal'], p['tool']['ev_lang']])

    for i, p in enumerate(temporal):
        dl1, dl2, id1, id2 = p['dl']
        if i < 6:
            q('temporal', 'What is the current deadline for %s?' % p['title'], [dl2], [id2], [p['ev_dl2']])
        else:
            q('temporal', 'What deadline was originally set for %s?' % p['title'], [dl1], [id1], [p['ev_dl1']])

    for n in absent_people:
        q('absent_entity', 'In what year did %s join the lab?' % n, [], [])
    for n in ABSENT_PROJECTS:
        q('absent_entity', 'Who leads Project %s?' % n, [], [])
    for n in ABSENT_TOOLS[:2]:
        q('absent_entity', 'Which language is %s written in?' % n, [], [])
    q('absent_entity', 'How many desks does the %s office have?' % ABSENT_SITES[0], [], [])

    for p in R.sample(people, 4):
        q('absent_attribute', "What is %s's phone extension?" % p['name'], [], [])
    for p in R.sample(people, 2):
        q('absent_attribute', "When is %s's birthday?" % p['name'], [], [])
    for tl in R.sample(tools, 2):
        q('absent_attribute', 'Under which license is %s released?' % tl['name'], [], [])
    for p in R.sample([x for x in projects if x not in budgeted], 3):
        q('absent_attribute', 'How many compute credits did %s get approved?' % p['title'], [], [])
    q('absent_attribute', 'Who owns the building of the %s office?' % R.choice(sites)['name'], [], [])

    for i, x in enumerate(qs):
        x['id'] = 'q%03d' % (i + 1)
        x['evidence'] = [plain(e) for e in x['evidence']]

    # --------------------------------------------------- self-checks: the gold must be true
    plain_notes = {k: plain(v) for k, v in notes.items()}
    blob = '\n'.join(plain_notes.values()).lower()
    for x in qs:
        for s in x['support']:
            assert s in notes, (x['id'], s)
        for e in x['evidence']:
            assert any(e in plain_notes[s] for s in x['support']), (x['id'], e)
        if x['answers']:
            assert any(a.lower() in ' '.join(x['evidence']).lower() for a in x['answers']), x['id']
            # an answer that is already in the question matches every sentence about the subject
            # word-bounded, like the scorer: 'Ann' inside 'Joanna' is not the answer leaking
            assert not any(re.search(r'(?<![a-z0-9])' + re.escape(a.lower()) + r'(?![a-z0-9])',
                                     x['question'].lower()) for a in x['answers']), x['id']
    for n in absent_people + ABSENT_PROJECTS + ABSENT_TOOLS + ABSENT_SITES:
        assert n.lower() not in blob, n
    for word in ('phone', 'birthday', 'license', 'licence', 'owns the building', 'landlord'):
        assert word not in blob, word
    for p in projects:
        if p not in budgeted:
            assert 'credits for project %s' % p['name'].lower() not in blob
    return notes, qs


def write(notes, qs, vault_dir, qfile):
    vault_dir = Path(vault_dir)
    if vault_dir.exists():
        for f in vault_dir.glob('*.md'):
            f.unlink()
    vault_dir.mkdir(parents=True, exist_ok=True)
    for nid, text in sorted(notes.items()):
        (vault_dir / (nid + '.md')).write_bytes(text.encode('utf-8'))
    Path(qfile).write_bytes(('\n'.join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in qs) + '\n').encode('utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='compare a fresh build with the committed files')
    a = ap.parse_args()
    notes, qs = build()
    if not a.check:
        write(notes, qs, HERE / 'vault', HERE / 'questions.jsonl')
        cls = {}
        for x in qs: cls[x['cls']] = cls.get(x['cls'], 0) + 1
        print('wrote %d notes -> bench/vault, %d questions -> bench/questions.jsonl %s' % (len(notes), len(qs), cls))
        return 0
    with tempfile.TemporaryDirectory() as td:
        write(notes, qs, Path(td) / 'vault', Path(td) / 'questions.jsonl')
        from bench_core import tree_sha256, file_sha256
        fresh = (tree_sha256(Path(td) / 'vault'), file_sha256(Path(td) / 'questions.jsonl'))
    committed = (tree_sha256(HERE / 'vault'), file_sha256(HERE / 'questions.jsonl'))
    if fresh != committed:
        print('DRIFT: committed vault/questions differ from make_vault.py output. '
              'Edit the generator, not the files, then rerun it.')
        return 1
    print('ok: committed vault and questions match the generator (vault %s, questions %s)' % (fresh[0][:12], fresh[1][:12]))
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    sys.exit(main())
