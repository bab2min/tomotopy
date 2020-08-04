'''
module for printing summary of topic models
'''

def _extract_param_desc(mdl_type:type):
    import re
    ps = mdl_type.__doc__.split('\nParameters\n')[1].split('\n')
    param_name = re.compile(r'^([a-zA-Z0-9_]+)\s*:\s*')
    directive = re.compile(r'^\s*\.\.')
    descriptive = re.compile(r'\s+([^\s].*)')
    period = re.compile(r'[.,](\s|$)')
    ret = {}
    name = None
    desc = ''
    for p in ps:
        if directive.search(p): continue
        m = param_name.search(p)
        if m:
            if name: ret[name] = desc.split('. ')[0]
            name = m.group(1)
            desc = ''
            continue
        m = descriptive.search(p)
        if m:
            desc += (' ' if desc else '') + m.group(1)
            continue
    if name: ret[name] = period.split(desc)[0]
    return ret

def _call_method_bound(mdl, method:str, *args, **kwargs):
    glob_methods = globals()
    for c in type(mdl).mro()[:-1]:
        cname = c.__name__
        try:
            return glob_methods[method + '_' + cname](mdl, *args, **kwargs)
        except KeyError:
            pass
    raise KeyError(method + '_' + cname)

def _format_numpy(arr, prefix=''):
    import numpy as np
    arr = np.array(arr)
    return ('\n' + prefix).join(str(arr).split('\n'))

def basic_info_LDAModel(mdl, file):
    import tomotopy as tp
    import numpy as np

    p = mdl.used_vocab_freq / mdl.used_vocab_freq.sum()
    entropy = (p * np.log(p)).sum()

    print('| {} (current version: {})'.format(type(mdl).__name__, tp.__version__), file=file)
    print('| {} docs, {} words'.format(len(mdl.docs), mdl.num_words), file=file)
    print('| Total Vocabs: {}, Used Vocabs: {}'.format(len(mdl.vocabs), len(mdl.used_vocabs)), file=file)
    print('| Entropy of words: {:.5f}'.format(entropy), file=file)
    print('| Removed Vocabs: {}'.format(' '.join(mdl.removed_top_words) if mdl.removed_top_words else '<NA>'), file=file)

def basic_info_DMRModel(mdl, file):
    from collections import Counter
    basic_info_LDAModel(mdl, file)
    md_cnt = Counter(doc.metadata for doc in mdl.docs)
    print('| Metadata of docs and its distribution', file=file)
    for md in mdl.metadata_dict:
        print('|  {}: {}'.format(md, md_cnt.get(md, 0)), file=file)

def basic_info_GDMRModel(mdl, file):
    import numpy as np
    basic_info_LDAModel(mdl, file)
    md_stack = np.stack([doc.metadata for doc in mdl.docs])
    md_min = md_stack.min(axis=0)
    md_max = md_stack.max(axis=0)
    md_avg = np.average(md_stack, axis=0)
    md_std = np.std(md_stack, axis=0)
    print('| Metadata distribution of docs', file=file)
    for i in range(md_stack.shape[1]):
        print('|  #{}: Range={:.5}~{:.5}, Avg={:.5}, Stdev={:.5}'.format(i, md_min[i], md_max[i], md_avg[i], md_std[i]), file=file)

def basic_info_LLDAModel(mdl, file):
    from collections import Counter
    basic_info_LDAModel(mdl, file)
    label_cnt = Counter(l for doc in mdl.docs for l, _ in doc.labels)
    print('| Label of docs and its distribution', file=file)
    for lb in mdl.topic_label_dict:
        print('|  {}: {}'.format(lb, label_cnt.get(lb, 0)), file=file)

def basic_info_PLDAModel(mdl, file):
    from collections import Counter
    basic_info_LDAModel(mdl, file)
    label_cnt = Counter(l for doc in mdl.docs for l, _ in doc.labels)
    print('| Label of docs and its distribution', file=file)
    for lb in mdl.topic_label_dict:
        print('|  {}: {}'.format(lb, label_cnt.get(lb, 0)), file=file)

def training_info_LDAModel(mdl, file):
    print('| Iterations: {}, Burn-in steps: {}'.format(mdl.global_step, mdl.burn_in), file=file)
    print('| Optimization Interval: {}'.format(mdl.optim_interval), file=file)
    print('| Log-likelihood per word: {:.5f}'.format(mdl.ll_per_word), file=file)

def initial_params_info_LDAModel(mdl, file):
    import tomotopy as tp
    param_desc = _extract_param_desc(type(mdl))
    if hasattr(mdl, '_init_params'):
        for k, v in mdl._init_params.items():
            if type(v) is float: fmt = ':.5'
            else: fmt = ''

            try:
                _call_method_bound(mdl, 'initial_params_info_' + k, v, file=file)
            except KeyError:
                if k in param_desc:
                    print(('| {}: {' + fmt + '} ({})').format(k, v, param_desc[k]), file=file)
                else:
                    print(('| {}: {' + fmt + '}').format(k, v), file=file)
    else:
        print('| Not Available (The model seems to have been built in version < 0.9.0.)', file=file)

def initial_params_info_tw_LDAModel(mdl, v, file):
    import tomotopy as tp
    print('| tw: TermWeight.{}'.format(tp.TermWeight(v).name), file=file)

def initial_params_info_version_LDAModel(mdl, v, file):
    import tomotopy as tp
    print('| trained in version {}'.format(v), file=file)


def initial_params_info_vars_SLDAModel(mdl, v, file):
    import tomotopy as tp
    var_type = {'l':'linear', 'b':'binary'}
    print('| vars: {}'.format(', '.join(map(var_type.__getitem__, v))), file=file)

def params_info_LDAModel(mdl, file):
    print('| alpha (Dirichlet prior on the per-document topic distributions)\n'
        '|  {}'.format(_format_numpy(mdl.alpha, '|  ')), file=file)
    print('| eta (Dirichlet prior on the per-topic word distribution)\n'
        '|  {:.5}'.format(mdl.eta), file=file)

def params_info_SLDAModel(mdl, file):
    params_info_LDAModel(mdl, file)
    var_type = {'l':'linear', 'b':'binary'}
    print('| regression coefficients of response variables', file=file)
    for f in range(mdl.f):
        print('|  #{} ({}): {}'.format(f, 
            var_type.get(mdl.get_var_type(f)),
            _format_numpy(mdl.get_regression_coef(f), '|    ')
        ), file=file)

def params_info_DMRModel(mdl, file):
    print('| lambda (feature vector per metadata of documents)\n'
        '|  {}'.format(_format_numpy(mdl.lambdas, '|  ')), file=file)
    print('| alpha (Dirichlet prior on the per-document topic distributions for each metadata)', file=file)
    for i, md in enumerate(mdl.metadata_dict):
        print('|  {}: {}'.format(md, _format_numpy(mdl.alpha[:, i], '|    ')), file=file)
    print('| eta (Dirichlet prior on the per-topic word distribution)\n'
        '|  {:.5}'.format(mdl.eta), file=file)

def params_info_GDMRModel(mdl, file):
    print('| lambda (feature vector per metadata of documents)\n'
        '|  {}'.format(_format_numpy(mdl.lambdas, '|  ')), file=file)
    print('| eta (Dirichlet prior on the per-topic word distribution)\n'
        '|  {:.5}'.format(mdl.eta), file=file)

def params_info_PAModel(mdl, file):
    print('| alpha (Dirichlet prior on the per-document super topic distributions)\n'
        '|  {}'.format(_format_numpy(mdl.alpha, '|  ')), file=file)
    print('| subalpha (Dirichlet prior on the sub topic distributions for each super topic)', file=file)
    for k1 in range(mdl.k1):
        print('|  Super #{}: {}'.format(k1, _format_numpy(mdl.subalpha[k1], '|   ')), file=file)
    print('| eta (Dirichlet prior on the per-subtopic word distribution)\n'
        '|  {:.5}'.format(mdl.eta), file=file)

def params_info_HPAModel(mdl, file):
    print('| alpha (Dirichlet prior on the per-document super topic distributions)\n'
        '|  {} {}'.format(mdl.alpha[:1], _format_numpy(mdl.alpha[1:], '|  ')), file=file)
    print('| subalpha (Dirichlet prior on the sub topic distributions for each super topic)', file=file)
    for k1 in range(mdl.k1):
        print('|  Super #{}: {} {}'.format(k1, mdl.subalpha[k1, :1], _format_numpy(mdl.subalpha[k1, 1:], '|   ')), file=file)
    print('| eta (Dirichlet prior on the per-subtopic word distribution)\n'
        '|  {:.5}'.format(mdl.eta), file=file)

def params_info_HDPModel(mdl, file):
    print('| alpha (concentration coeficient of Dirichlet Process for document-table)\n'
        '|  {:.5}'.format(mdl.alpha), file=file)
    print('| eta (Dirichlet prior on the per-topic word distribution)\n'
        '|  {:.5}'.format(mdl.eta), file=file)
    print('| gamma (concentration coeficient of Dirichlet Process for table-topic)\n'
        '|  {:.5}'.format(mdl.gamma), file=file)
    print('| Number of Topics: {}'.format(mdl.live_k), file=file)
    print('| Number of Tables: {}'.format(mdl.num_tables), file=file)

def params_info_HLDAModel(mdl, file):
    print('| alpha (Dirichlet prior on the per-document depth level distributions)\n'
        '|  {}'.format(_format_numpy(mdl.alpha, '|  ')), file=file)
    print('| eta (Dirichlet prior on the per-topic word distribution)\n'
        '|  {:.5}'.format(mdl.eta), file=file)
    print('| gamma (concentration coeficient of Dirichlet Process)\n'
        '|  {:.5}'.format(mdl.gamma), file=file)
    print('| Number of Topics: {}'.format(mdl.live_k), file=file)

def params_info_DTModel(mdl, file):
    print('| alpha (Dirichlet prior on the per-document topic distributions for each timepoint)\n'
        '|  {}'.format(_format_numpy(mdl.alpha, '|  ')), file=file)
    print('| phi (Dirichlet prior on the per-time&topic word distribution)\n'
        '|  ...', file=file)

def topics_info_LDAModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_topics()
    for k in range(mdl.k):
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('| #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)

def topics_info_HDPModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_topics()
    for k in range(mdl.k):
        if not mdl.is_live_topic(k): continue
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('| #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)

def topics_info_HLDAModel(mdl, file, topic_word_top_n):
    import numpy as np
    topic_cnt = mdl.get_count_by_topics()

    def print_hierarchical(k=0, level=0):
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('| {}#{} ({}) : {}'.format('  ' * level, k, topic_cnt[k], words), file=file)
        for c in np.sort(mdl.children_topics(k)):
            print_hierarchical(c, level + 1)

    print_hierarchical()

def topics_info_PAModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_super_topic()
    print('| Sub-topic distribution of Super-topics', file=file)
    for k in range(mdl.k1):
        words = ' '.join('#{}'.format(w) for w, _ in mdl.get_sub_topics(k, top_n=topic_word_top_n))
        print('|  #Super{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)
    topic_cnt = mdl.get_count_by_topics()
    print('| Word distribution of Sub-topics', file=file)
    for k in range(mdl.k2):
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('|  #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)

def topics_info_HPAModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_topics()
    words = ' '.join(w for w, _ in mdl.get_topic_words(0, top_n=topic_word_top_n))
    print('| Top-topic ({}) : {}'.format(topic_cnt[0], words), file=file)
    print('| Super-topics', file=file)
    for k in range(1, 1 + mdl.k1):
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('|  #Super{} ({}) : {}'.format(k - 1, topic_cnt[k], words), file=file)
        words = ' '.join('#{}'.format(w) for w, _ in mdl.get_sub_topics(k - 1, top_n=topic_word_top_n))
        print('|    its sub-topics : {}'.format(words))
    print('| Sub-topics', file=file)
    for k in range(1 + mdl.k1, 1 + mdl.k1 + mdl.k2):
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('|  #{} ({}) : {}'.format(k - 1 - mdl.k1, topic_cnt[k], words), file=file)

def topics_info_LLDAModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_topics()
    for k in range(mdl.k):
        label = ('Label {} (#{})'.format(mdl.topic_label_dict[k], k) 
            if k < len(mdl.topic_label_dict) else '#{}'.format(k))
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('| {} ({}) : {}'.format(label, topic_cnt[k], words), file=file)

def topics_info_PLDAModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_topics()
    for k in range(mdl.k):
        l = k // mdl.topics_per_label
        label = ('Label {}-{} (#{})'.format(mdl.topic_label_dict[l], k % mdl.topics_per_label, k) 
            if l < len(mdl.topic_label_dict) else 'Latent {} (#{})'.format(k - mdl.topics_per_label * len(mdl.topic_label_dict), k))
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('| {} ({}) : {}'.format(label, topic_cnt[k], words), file=file)

def topics_info_MGLDAModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_topics()
    print('| Global Topic', file=file)
    for k in range(mdl.k):
        words = ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=topic_word_top_n))
        print('|  #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)
    print('| Local Topic', file=file)
    for k in range(mdl.k_l):
        words = ' '.join(w for w, _ in mdl.get_topic_words(k + mdl.k, top_n=topic_word_top_n))
        print('|  #{} ({}) : {}'.format(k, topic_cnt[k + mdl.k], words), file=file)

def topics_info_DTModel(mdl, file, topic_word_top_n):
    topic_cnt = mdl.get_count_by_topics()
    for k in range(mdl.k):
        print('| #{} ({})'.format(k, topic_cnt[:, k].sum()), file=file)
        for t in range(mdl.num_timepoints):
            words = ' '.join(w for w, _ in mdl.get_topic_words(k, t, top_n=topic_word_top_n))
            print('|  t={} ({}) : {}'.format(t, topic_cnt[t, k], words), file=file)

def summary(mdl, initial_hp=True, params=True, topic_word_top_n=5, file=None, flush=False):
    import tomotopy as tp
    import numpy as np
    import sys

    file = file or sys.stdout
    flush = flush or False

    print('<Basic Info>', file=file)
    _call_method_bound(mdl, 'basic_info', file=file)
    print('|', file=file)
    print('<Training Info>', file=file)
    _call_method_bound(mdl, 'training_info', file=file)
    print('|', file=file)

    if initial_hp:
        print('<Initial Parameters>', file=file)
        _call_method_bound(mdl, 'initial_params_info', file=file)
        print('|', file=file)
    
    if params:
        print('<Parameters>', file=file)
        _call_method_bound(mdl, 'params_info', file=file)
        print('|', file=file)

    if topic_word_top_n > 0:
        print('<Topics>', file=file)
        _call_method_bound(mdl, 'topics_info', file=file, topic_word_top_n=topic_word_top_n)
        print('|', file=file)

    print(file=file, flush=flush)
