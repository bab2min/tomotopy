import os
import re
import io
import urllib.parse
import html
import math
import json
import csv
import io
import functools
from collections import defaultdict, Counter
from dataclasses import dataclass
import traceback
import http.server

import numpy as np

def _escape_html(s):
    if isinstance(s, str):
        return html.escape(s)
    else:
        return s

@dataclass
class Document:
    doc_id: int
    topic_dist: list
    words: list = None
    metadata: dict = None

@dataclass
class Topic:
    topic_id: int
    words: list

def hue2rgb(h:float, b:float = 1):
    h = h % 6.0
    b = min(max(b, 0), 1)
    if h < 1:
        return (int(255 * b), int(255 * h * b), 0)
    elif h < 2:
        return (int(255 * (2 - h) * b), int(255 * b), 0)
    elif h < 3:
        return (0, int(255 * b), int(255 * (h - 2) * b))
    elif h < 4:
        return (0, int(255 * (4 - h) * b), int(255 * b))
    elif h < 5:
        return (int(255 * (h - 4) * b), 0, int(255 * b))
    else:
        return (int(255 * b), 0, int(255 * (6 - h) * b))

def scale_color(s:float, scale='log'):
    if scale == 'log':
        s = min(max(s, 0) + 1e-4, 1)
        s = max(math.log(s) + 4, 0) * (6 / 4)
    else:
        s = min(max(s, 0), 1)
        s *= 6
    if s < 1:
        return hue2rgb(4, s * 0.6)
    elif s < 5:
        return hue2rgb(5 - s, (s - 1) / 4 * 0.4 + 0.6)
    else:
        t = int((s - 5) * 255)
        return (255, t, t)

def colorize(a, colors):
    a = a * (len(colors) - 1)
    l = np.floor(a).astype(np.int32)
    r = np.clip(l + 1, 0, len(colors) - 1)
    a = a - l
    result = colors[l] + (colors[r] - colors[l]) * a[..., None]
    return result

def draw_contour_map(arr, interval, smooth=True):
    def _refine(a, smooth, cut=0.15, scale=0.6):
        if smooth:
            s = np.zeros_like(a, dtype=np.float32)
            a = np.pad(a, (2, 2), 'edge')

            # approximated 5x5 gaussian filter

            s += a[2:-2, 2:-2] * 41

            s += a[1:-3, 2:-2] * 26
            s += a[3:-1, 2:-2] * 26
            s += a[2:-2, 1:-3] * 26
            s += a[2:-2, 3:-1] * 26

            s += a[1:-3, 1:-3] * 16
            s += a[3:-1, 3:-1] * 16
            s += a[1:-3, 3:-1] * 16
            s += a[3:-1, 1:-3] * 16

            s += a[4:, 2:-2] * 7
            s += a[:-4, 2:-2] * 7
            s += a[2:-2, 4:] * 7
            s += a[2:-2, :-4] * 7
            
            s += a[1:-3, 4:] * 4
            s += a[3:-1, :-4] * 4
            s += a[4:, 3:-1] * 4
            s += a[:-4, 1:-3] * 4

            s += a[4:, 4:] * 1
            s += a[:-4, :-4] * 1
            s += a[4:, :-4] * 1
            s += a[:-4, 4:] * 1

            s /= 273

            a = (s - cut) / scale
        return a.clip(0, 1)

    cv = np.floor(arr / interval)
    contour_map = np.zeros_like(arr, dtype=np.float32)
    contour_map[:-1] += cv[:-1] != cv[1:]
    contour_map[1:] += cv[:-1] != cv[1:]
    contour_map[:, :-1] += cv[:, :-1] != cv[:, 1:]
    contour_map[:, 1:] += cv[:, :-1] != cv[:, 1:]
    contour_map = _refine(contour_map, smooth, cut=0.25, scale=0.65)

    cv = np.floor(arr / (interval * 5))
    contour_map2 = np.zeros_like(arr, dtype=np.float32)
    contour_map2[:-1] += cv[:-1] != cv[1:]
    contour_map2[1:] += cv[:-1] != cv[1:]
    contour_map2[:, :-1] += cv[:, :-1] != cv[:, 1:]
    contour_map2[:, 1:] += cv[:, :-1] != cv[:, 1:]
    contour_map2 = _refine(contour_map2, smooth, cut=0.15, scale=0.6)
    contour_map = (contour_map + contour_map2) / 2

    return contour_map

topic_colors = [hue2rgb((i * 7) / 40 * 6) for i in range(40)]
topic_styles = [(i * 7 // 40) % 3 for i in range(40)]

def find_best_labels_for_range(start, end, max_labels):
    dist = end - start
    unit = 1
    for i in range(10):
        u = 10 ** (-i)
        if dist < u:
            continue
        r = dist - round(dist / u) * u
        if abs(r) < u * 0.1:
            unit = u
            break
    steps = round(dist / unit)
    while steps > max_labels:
        unit *= 10
        steps = round(dist / unit)

    if steps <= max_labels / 5:
        unit /= 5
        steps = round(dist / unit)
    elif steps <= max_labels / 2:
        unit /= 2
        steps = round(dist / unit)
    
    s = int(math.floor(start / unit))
    e = int(math.ceil(end / unit))
    return [unit * i for i in range(s, e + 1)]

def estimate_confidence_interval_of_dd(alpha, p=0.95, samples=16384):
    rng = np.random.RandomState(0)
    alpha = np.array(alpha, dtype=np.float32)
    mean = alpha / alpha.sum()
    t = rng.dirichlet(alpha, samples).astype(np.float32)
    t.sort(axis=0)
    cnt = int(samples * (1 - p))
    i = (t[-cnt:] - t[:cnt]).argmin(0)

    o = np.array([np.searchsorted(t[:, i], m, 'right') for i, m in enumerate(mean)])
    i = np.maximum(i, o - (samples - cnt))

    lb = t[i, np.arange(len(alpha))]
    ub = t[i - cnt, np.arange(len(alpha))]
    return lb, ub
 
def is_iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False

class DocumentFilter:
    def __init__(self, model, max_cache_size=5) -> None:
        self.model = model
        self.max_cache_size = max_cache_size
        self._cached = {}
        self._cached_keys = []
    
    def _sort_and_filter(self, sort_key:int, filter_target:int, filter_value:float, filter_keyword:tuple, filter_metadata:str):
        results = []
        for i, doc in enumerate(self.model.docs):
            if filter_keyword and not all(kw in doc.raw.lower() for kw in filter_keyword): continue
            if filter_metadata is not None and doc.metadata != filter_metadata: continue
            dist = doc.get_topic_dist()
            if dist[filter_target] < filter_value: continue
            if sort_key >= 0:
                results.append((dist[sort_key], i))
            else:
                results.append(i)

        if sort_key < 0:
            return results
        else:
            return [i for _, i in sorted(results, reverse=True)]

    def _get_cached_filter_result(self, sort_key:int, filter_target:int, filter_value:float, filter_keyword:str, filter_metadata:str):
        filter_keyword = tuple(filter_keyword.lower().split())
        if sort_key < 0 and filter_value <= 0 and not filter_keyword and filter_metadata is None:
            # return None for no filtering nor sorting
            return None
        key = (sort_key, filter_target, filter_value, filter_keyword, filter_metadata)
        if key in self._cached:
            return self._cached[key]
        else:
            result = self._sort_and_filter(sort_key, filter_target, filter_value, filter_keyword, filter_metadata)
            if len(self._cached_keys) >= self.max_cache_size:
                del self._cached[self._cached_keys.pop(0)]
            self._cached[key] = result
            self._cached_keys.append(key)
            return result

    def get(self, sort_key:int, filter_target:int, filter_value:float, filter_keyword:str, filter_metadata:str, index:slice):
        # return (doc_indices, total_docs_filtered)
        result = self._get_cached_filter_result(sort_key, filter_target, filter_value, filter_keyword, filter_metadata)
        if result is None:
            return list(range(index.start, min(index.stop, len(self.model.docs)))), len(self.model.docs)
        else:
            return result[index], len(result)


class ViewerHandler(http.server.SimpleHTTPRequestHandler):

    get_handlers = [
        (r'/?', 'overview'),
        (r'/document/?', 'document'),
        (r'/topic/?', 'topic'),
        (r'/topic-rel/?', 'topic_rel'),
        (r'/metadata/?', 'metadata'),
        (r'/tdf-map/?', 'tdf_map'),
        (r'/api/document/(\d+)', 'api_document'),
        (r'/api/conf-interval/(\d+)/([0-9.]+)', 'api_conf_interval'),
        (r'/d/topic-words\.csv', 'download_topic_words'),
        (r'/d/document-top1-topic\.csv', 'download_document_top1_topic'),
        (r'/d/tdf-map-([0-9]+|legend).png', 'download_tdf_map'),
    ]

    post_handlers = [
        (r'/api/topic/(\d+)/label', 'api_update_topic_label'),
    ]

    num_docs_per_page = 30

    @property
    def title(self):
        return self.server.title
    
    @property
    def model(self):
        return self.server.model
    
    @property
    def root_path(self):
        return self.server.root_path

    @property
    def model_hash(self):
        hex_chr = hex(self.model.get_hash())[2:]
        if len(hex_chr) < 32:
            hex_chr = '0' * (32 - len(hex_chr)) + hex_chr
        return hex_chr

    @property
    def available(self):
        ret = {}
        if 'GDMR' in type(self.model).__name__:
            ret['metadata'] = True
        return ret
    
    @property
    def tomotopy_version(self):
        import tomotopy
        return tomotopy.__version__

    @property
    def user_config(self):
        return self.get_user_config(None)
    
    @property
    def read_only(self):
        return self.server.read_only

    def get_topic_label(self, k, prefix='', id_suffix=False):
        label = self.get_user_config(('topic_label', k))
        if label is None:
            label = f'{prefix}#{k}'
        elif id_suffix:
            label += f' #{k}'
        return label

    def get_all_topic_labels(self, prefix='', id_suffix=False):
        return [self.get_topic_label(k, prefix, id_suffix) for k in range(self.model.k)]

    def get_user_config(self, key):
        if self.server.user_config is None:
            if self.server.user_config_file:
                try:
                    self.server.user_config = json.load(open(self.server.user_config_file, 'r', encoding='utf-8'))
                    model_hash_in_config = self.server.user_config.get('model_hash')
                    if model_hash_in_config is not None and model_hash_in_config != self.model_hash:
                        print(f'User config file is for a different model. Ignoring the file.')
                        self.server.user_config = {}
                except FileNotFoundError:
                    self.server.user_config = {}
            else:
                self.server.user_config = {}
            self.server.user_config['model_hash'] = self.model_hash
        if key is None:
            return self.server.user_config
        
        if isinstance(key, str) or not is_iterable(key):
            key = [key]
        
        obj = self.server.user_config
        for k in key:
            obj = obj.get(str(k))
            if obj is None:
                return obj
        return obj
    
    def set_user_config(self, key, value):
        self.get_user_config(key)

        if isinstance(key, str) or not is_iterable(key):
            key = [key]

        obj = self.server.user_config
        for k in key[:-1]:
            k = str(k)
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        obj[str(key[-1])] = value

        if self.server.user_config_file:
            json.dump(self.server.user_config, open(self.server.user_config_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    def render(self, **kwargs):
        local_vars = {}
        for k in dir(self):
            if k.startswith('_'): continue
            local_vars[k] = getattr(self, k)
        local_vars.update(kwargs)
        ret = []
        local_vars['ret'] = ret
        exec(self.server.template, None, local_vars)
        output = ''.join(ret)
        self.wfile.write(output.encode())
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path, query = parsed.path, parsed.query
        self.arguments = {k:v[0] if len(v) == 1 else v for k, v in urllib.parse.parse_qs(query).items()}
        for pattern, handler in self.get_handlers:
            m = re.fullmatch(pattern, path)
            if m:
                try:
                    getattr(self, 'get_' + handler)(*m.groups())
                except:
                    self.send_error(500)
                    traceback.print_exc()
                    self.wfile.write(traceback.format_exc().encode())
                return
        self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path, query = parsed.path, parsed.query
        self.arguments = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        for pattern, handler in self.post_handlers:
            m = re.fullmatch(pattern, path)
            if m:
                try:
                    getattr(self, 'post_' + handler)(*m.groups())
                except:
                    self.send_error(500)
                    traceback.print_exc()
                    self.wfile.write(traceback.format_exc().encode())
                return
        self.send_error(404)

    def get_overview(self):
        import tomotopy._summary as tps
        buf = io.StringIO()
        tps.basic_info(self.model, buf)
        basic_info = buf.getvalue()
        
        buf = io.StringIO()
        tps.training_info(self.model, buf)
        training_info = buf.getvalue()
        
        init_param_desc = tps._extract_param_desc(self.model)

        buf = io.StringIO()
        tps.params_info(self.model, buf)
        params_info = buf.getvalue()

        buf = io.StringIO()
        tps.topics_info(self.model, buf)
        topics_info = buf.getvalue()

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.render(action='overview',
                    basic_info=basic_info,
                    training_info=training_info,
                    init_param_desc=init_param_desc,
                    params_info=params_info,
                    topics_info=topics_info,
                    )
    
    def get_document(self):
        total_docs = len(self.model.docs)
        sort_key = int(self.arguments.get('s', '-1'))
        filter_target = int(self.arguments.get('t', '0'))
        filter_value = float(self.arguments.get('v', '0'))
        filter_keyword = self.arguments.get('sq', '')
        filter_metadata = int(self.arguments.get('m', '-1'))
        page = int(self.arguments.get('p', '0'))
        
        if not self.available.get('metadata') or filter_metadata < 0:
            filter_metadata = -1
            md = None
        else:
            md = self.model.metadata_dict[filter_metadata]

        doc_indices, filtered_docs = self.server.filter.get(
            sort_key, 
            filter_target, 
            filter_value / 100, 
            filter_keyword,
            md,
            slice(page * self.num_docs_per_page, (page + 1) * self.num_docs_per_page)
        )
        total_pages = (filtered_docs + self.num_docs_per_page - 1) // self.num_docs_per_page

        documents = []
        for i in doc_indices:
            documents.append(Document(i, self.model.docs[i].get_topic_dist()))
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.render(action='document',
                    page=page,
                    total_pages=total_pages,
                    filtered_docs=filtered_docs if filter_value > 0 or filter_keyword or filter_metadata >= 0 else None,
                    total_docs=total_docs,
                    documents=documents,
                    sort_key=sort_key,
                    filter_target=filter_target,
                    filter_value=filter_value,
                    filter_keyword=filter_keyword,
                    filter_metadata=filter_metadata,
                    )
    
    def prepare_topic_doc_stats(self):
        all_cnt = Counter([doc.get_topics(1)[0][0] for doc in self.model.docs])
        top1_topic_dist = [all_cnt[i] for i in range(self.model.k)]

        try:
            has_metadata = len(self.model.docs[0].metadata) > 1
        except:
            has_metadata = False
        
        if has_metadata:
            top1_topic_dist_by_metadata = defaultdict(Counter)
            for doc in self.model.docs:
                top1_topic_dist_by_metadata[doc.metadata][doc.get_topics(1)[0][0]] += 1
            for k, cnt in top1_topic_dist_by_metadata.items():
                top1_topic_dist_by_metadata[k] = [cnt[i] for i in range(self.model.k)]

        return top1_topic_dist, top1_topic_dist_by_metadata if has_metadata else None

    def get_topic(self):
        top_n = int(self.arguments.get('top_n', '10'))
        alpha = float(self.arguments.get('alpha', '0.0'))
        topics = []
        max_dist = 0
        if alpha > 0:
            topic_word_dist = np.stack([self.model.get_topic_word_dist(k) for k in range(self.model.k)])
            pseudo_idf = np.log(len(topic_word_dist) / (topic_word_dist ** alpha).sum(0))
            weighted_topic_word_dist = topic_word_dist * pseudo_idf
            top_words = (-weighted_topic_word_dist).argsort()[:, :top_n]
            max_dist = topic_word_dist.max()
            for k, top_word in enumerate(top_words):
                topic_words = [(self.model.vocabs[w], w, topic_word_dist[k, w]) for w in top_word]
                topics.append(Topic(k, topic_words))
        else:
            for k in range(self.model.k):
                topic_words = self.model.get_topic_words(k, top_n, return_id=True)
                max_dist = max(max_dist, topic_words[0][-1])
                topics.append(Topic(k, topic_words))
        
        top1_topic_dist, top1_topic_dist_by_metadata = self.prepare_topic_doc_stats()

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.render(action='topic',
                    topics=topics,
                    max_dist=max_dist,
                    top_n=top_n,
                    top1_topic_dist=top1_topic_dist,
                    top1_topic_dist_by_metadata=top1_topic_dist_by_metadata,
                    )

    def get_topic_rel(self):
        topic_word_dist = np.stack([self.model.get_topic_word_dist(k) for k in range(self.model.k)])
        overlaps = np.minimum(topic_word_dist[:, None], topic_word_dist[None]).sum(-1)
        similar_pairs = np.stack(np.unravel_index((-np.triu(overlaps, 1)).flatten().argsort(), overlaps.shape), -1)
        similar_pairs = similar_pairs[similar_pairs[:, 0] != similar_pairs[:, 1]]
        most_similars = (2 * np.eye(len(overlaps)) - overlaps).argsort()[:, :-1]

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.render(action='topic-rel',
                    overlaps=overlaps,
                    similar_pairs=similar_pairs,
                    most_similars=most_similars,
                    )

    def prepare_metadata(self):
        axis = int(self.arguments.get('axis', '0'))
        x = self.arguments.get('x', '')
        resolution = int(self.arguments.get('r', '33'))
        numeric_metadata = self.model.metadata_range
        if axis < 0 or axis >= len(numeric_metadata):
            axis = 0

        if x:
            x = list(map(float, x.split(',')))
            x = list(zip(x[::2], x[1::2]))
        else:
            x = [((s, e) if i == axis else (s, s)) for i, (s, e) in enumerate(numeric_metadata)]
        
        start, end = zip(*x)
        num = [resolution if i == axis else 1 for i in range(len(x))]
        squeeze_axis = tuple(i for i in range(len(x)) if i != axis)
        return start, end, num, squeeze_axis, axis, numeric_metadata

    def compute_data_density(self, x_values, axis, categorical_metadata):
        dist = defaultdict(list)
        for d in self.model.docs:
            dist[d.metadata].append(d.numeric_metadata[axis])
        
        s, e = self.model.metadata_range[axis]
        kernel_size = (e - s) / (self.model.degrees[axis] + 1)

        densities = []
        for c in categorical_metadata:
            points = np.array(dist[c], dtype=np.float32)
            density = np.exp(-((x_values[:, None] - points) / kernel_size) ** 2).sum(-1)
            density /= density.max()
            densities.append(density)
        return densities

    def get_metadata(self):
        (start, end, num, squeeze_axis, axis, numeric_metadata
         ) = self.prepare_metadata()
        max_labels = int(self.arguments.get('max_labels', '15'))
        categorical_metadata = self.model.metadata_dict
        
        x_values = np.linspace(start[axis], end[axis], num[axis], dtype=np.float32)
        data_density = self.compute_data_density(x_values, axis, categorical_metadata)
        boundaries = np.array(find_best_labels_for_range(x_values[0], x_values[-1], max_labels))
        t = (np.searchsorted(boundaries, x_values, 'right') - 1).clip(0)
        x_labels = [f'{boundaries[t[i]]:g}' if i == 0 or t[i - 1] != t[i] else '' for i in range(len(x_values))]

        cats = np.stack([self.model.tdf_linspace(start, end, num, metadata=c).squeeze(squeeze_axis) for c in categorical_metadata])
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.render(action='metadata',
                    categorical_metadata=categorical_metadata,
                    numeric_metadata=numeric_metadata,
                    range_start=start,
                    range_end=end,
                    x_values=x_values,
                    x_labels=x_labels,
                    axis=axis,
                    cats=cats,
                    data_density=data_density,
                    )

    def get_tdf_map(self):
        x = int(self.arguments.get('x', '0'))
        y = int(self.arguments.get('y', '1'))
        width = int(self.arguments.get('w', '640'))
        height = int(self.arguments.get('h', '480'))
        contour_interval = float(self.arguments.get('s', '0.2'))
        smooth = bool(int(self.arguments.get('smooth', '1')))
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.render(action='tdf-map',
                    x_axis=x,
                    y_axis=y,
                    width=width,
                    height=height,
                    contour_interval=contour_interval,
                    smooth=smooth,)

    def get_api_document(self, doc_id):
        doc_id = int(doc_id)
        doc = self.model.docs[doc_id]
        chunks = []
        raw = doc.raw
        last = 0
        for (s, e), topic, w in zip(doc.span, doc.topics, doc.words):
            if topic < 0: continue
            if e <= last: continue
            if last < s:
                chunks.append(html.escape(raw[last:s]))
            chunks.append(f'<span class="topic-color-{topic % 40}-040 topic-action" data-topic-id="{topic}" data-bs-toggle="tooltip" data-bs-title="Topic #{topic}: {self.model.vocabs[w]}">{html.escape(raw[s:e])}</span>')
            last = e
        if last < len(raw):
            chunks.append(html.escape(raw[last:]))
        html_cont = '<p>' + ''.join(chunks).strip().replace('\n', '<br>') + '</p>'
        
        meta = []
        if hasattr(doc, 'metadata'):
            meta.append(f'<meta name="metadata" content="{html.escape(doc.metadata)}">')
        if hasattr(doc, 'multi_metadata'):
            meta.append(f'<meta name="multi_metadata" content="{html.escape(repr(doc.multi_metadata))}">')
        if hasattr(doc, 'numeric_metadata'):
            meta.append(f'<meta name="numeric_metadata" content="{html.escape(repr(doc.numeric_metadata.tolist()))}">')
        if meta:
            html_cont = '\n'.join(meta) + '\n' + html_cont

        chunks = ['<table class="table table-striped table-hover topic-dist">']
        for topic_id, dist in doc.get_topics(top_n=-1):
            chunks.append(
f'''<tr class="topic-action" data-topic-id="{topic_id}">
    <th scope="row"><a href="javascript:void(0);">{self.get_topic_label(topic_id, prefix="Topic ", id_suffix=True)}</a></th>
    <td><div class="progress flex-grow-1" data-bs-toggle="tooltip" data-bs-title="{dist:.5%}">
        <div class="progress-bar topic-color-{topic_id % 40}" role="progressbar" style="width: {dist:.5%}">{dist:.3%}</div>
    </div></td>
    </tr>''')
        chunks.append('</table>')
        html_cont += ''.join(chunks)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_cont.encode())

    def get_api_conf_interval(self, cid, p=0.95):
        cid = int(cid)
        p = float(p)
        (start, end, num, squeeze_axis, axis, numeric_metadata
         ) = self.prepare_metadata()
        categorical_metadata = self.model.metadata_dict
        alphas = np.exp(self.model.tdf_linspace(start, end, num, metadata=categorical_metadata[cid], normalize=False).squeeze(squeeze_axis))
        lbs = []
        ubs = []
        for alpha in alphas:
            lb, ub = estimate_confidence_interval_of_dd(alpha, p=p, samples=10000)
            lbs.append(lb)
            ubs.append(ub)
        lbs = np.stack(lbs, axis=-1).tolist()
        ubs = np.stack(ubs, axis=-1).tolist()

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        self.wfile.write(json.dumps({'data':{'cid':cid, 'p':p, 'lbs': lbs, 'ubs': ubs}}, ensure_ascii=False).encode())

    def post_api_update_topic_label(self, topic_id):
        if self.read_only:
            self.send_error(403)
            return
        topic_id = int(topic_id)
        label = self.arguments.get('label', '') or None
        self.set_user_config(('topic_label', topic_id), label)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'topic_id':topic_id, 'label':label}, ensure_ascii=False).encode())

    def get_download_topic_words(self):
        n = int(self.arguments.get('n', '10'))
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        headers = ['']
        words = []
        for k in range(self.model.k):
            headers.append(self.get_topic_label(k, prefix='Topic ', id_suffix=True))
            headers.append('Prob.')
            words.append(self.model.get_topic_words(k, top_n=n))

        writer.writerow(headers)
        for i in range(n):
            row = [i + 1]
            for k in range(self.model.k):
                row.extend(words[k][i])
            writer.writerow(row)

        self.send_response(200)
        self.send_header('Content-type', 'text/csv')
        self.send_header('Content-Disposition', 'attachment; filename="topic-words.csv"')
        self.end_headers()
        self.wfile.write(csv_buf.getvalue().encode('utf-8-sig'))

    def get_download_document_top1_topic(self):
        metadata = int(self.arguments.get('m', '0'))
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        top1_topic_dist, top1_topic_dist_by_metadata = self.prepare_topic_doc_stats()
        if metadata:
            headers = ['', *self.model.metadata_dict]
            writer.writerow(headers)
            for k in range(self.model.k):
                row = [self.get_topic_label(k, prefix='Topic ', id_suffix=True)]
                for m in self.model.metadata_dict:
                    row.append(top1_topic_dist_by_metadata[m][k])
                writer.writerow(row)
        else:
            headers = ['', 'All']
            writer.writerow(headers)
            for k, cnt in enumerate(top1_topic_dist):
                writer.writerow([self.get_topic_label(k, prefix='Topic ', id_suffix=True), cnt])
        
        self.send_response(200)
        self.send_header('Content-type', 'text/csv')
        if metadata:
            self.send_header('Content-Disposition', 'attachment; filename="document-top1-topic-by-metadata.csv"')
        else:
            self.send_header('Content-Disposition', 'attachment; filename="document-top1-topic.csv"')
        self.end_headers()
        self.wfile.write(csv_buf.getvalue().encode('utf-8-sig'))

    def __eq__(self, other):
        return self.model.get_hash() == other.model.get_hash()

    def __hash__(self):
        return self.model.get_hash()

    @functools.lru_cache(maxsize=128)
    def cached_tdf_linspace(self, start, end, num, metadata=""):
        return self.model.tdf_linspace(start, end, num, metadata=metadata)

    @functools.lru_cache(maxsize=128)
    def cache_tdf_map_img(self, topic_id, x, y, w, h, r, contour_interval, smooth):
        from PIL import Image
        start, end = zip(*r)
        num = [1] * len(start)
        num[x] = w
        num[y] = h

        metadata = int(self.arguments.get('m', '0'))
        metadata = self.model.metadata_dict[metadata] if metadata >= 0 else ""

        td = self.cached_tdf_linspace(tuple(start), tuple(end), tuple(num), metadata)
        td = td.transpose([-1, y, x] + [i for i in range(len(start)) if i not in (x, y)]).squeeze()
        max_val = np.log(td.max() + 1e-9) - 1
        min_val = -7
        if topic_id == 'legend':
            logits = np.linspace(min_val, max_val, w)[None]
            logits = np.repeat(logits, h, 0)
            smooth = False
        else:
            logits = np.log(td[topic_id] + 1e-9)
            logits = logits[::-1] # invert y-axis
        scaled = (logits - min_val) / (max_val - min_val)
        scaled = np.clip(scaled, 0, 1)

        contour_map = draw_contour_map(logits, contour_interval, smooth)

        colors = np.array([
            [0, 0, 0],
            [0, 0, 0.7],
            [0, 0.75, 0.75],
            [0, 0.8, 0],
            [0.85, 0.85, 0],
            [1, 0, 0],
            [1, 1, 1],
        ], dtype=np.float32)
        colorized = colorize(scaled, colors)
        if topic_id == 'legend':
            is_sub_grid = contour_map[0] < 1
            contour_map[:-int(h * 0.32)] = 0
            contour_map[:-int(h * 0.16), is_sub_grid] = 0
            contour_map = contour_map.clip(0, 1)
        colorized *= 1 - contour_map[..., None]
        img = Image.fromarray((colorized * 255).astype(np.uint8), 'RGB')
        img_buf = io.BytesIO()
        img.save(img_buf, format='PNG')
        img_buf.seek(0)
        return img_buf.read()

    def get_download_tdf_map(self, topic_id):
        if not hasattr(self.model, 'tdf_linspace'):
            self.send_error(404)
            return
        if topic_id == 'legend':
            pass
        else:
            topic_id = int(topic_id)
            if topic_id >= self.model.k:
                self.send_error(404)
                return
        
        x = int(self.arguments.get('x', '0'))
        y = int(self.arguments.get('y', '1'))
        w = int(self.arguments.get('w', '640'))
        h = int(self.arguments.get('h', '480'))
        contour_interval = float(self.arguments.get('s', '0.2'))
        smooth = bool(int(self.arguments.get('smooth', '1')))

        r = self.arguments.get('r', '')
        if r:
            r = list(map(float, x.split(',')))
            r = list(zip(x[::2], x[1::2]))
        else:
            r = self.model.metadata_range

        img_buf = self.cache_tdf_map_img(topic_id, x, y, w, h, tuple(r), contour_interval, smooth)
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        self.wfile.write(img_buf)

def _repl(m):
    if m.group().startswith('{{'):
        inner = m.group(1)
        if re.search(r':[-+0-9.a-z%]+$', inner):
            return '{' + inner + '}'
        return '{_escape_html(' + inner + ')}'
    elif m.group() == '{':
        return '{{'
    elif m.group() == '}':
        return '}}'

def _prepare_template():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    template = open(os.path.join(cur_dir, 'template.html'), 'r', encoding='utf-8').read()
    codes = []
    chunks = re.split(r'\{% *(.+?) *%\}', template)
    chunks.append(None)
    indentation = 0
    for text, cmd in zip(chunks[::2], chunks[1::2]):
        text = re.sub(r'\{\{(.+?)\}\}|[{}]', _repl, text)
        text = 'f' + repr(text)
        codes.append(' ' * indentation + 'ret.append(' + text + ')')
        if not cmd: continue
        if cmd.startswith('if '):
            codes.append(' ' * indentation + cmd + ':')
            indentation += 1
        elif cmd.startswith('elif '):
            indentation -= 1
            codes.append(' ' * indentation + cmd + ':')
            indentation += 1
        elif cmd.startswith('else'):
            indentation -= 1
            codes.append(' ' * indentation + cmd + ':')
            indentation += 1
        elif cmd.startswith('for '):
            codes.append(' ' * indentation + cmd + ':')
            indentation += 1
        elif cmd == 'end':
            indentation -= 1
        elif cmd.startswith('set '):
            codes.append(' ' * indentation + cmd[4:])
    compiled_template = compile('\n'.join(codes), 'template.html', 'exec')
    return compiled_template

def open_viewer(model, host='localhost', port=80, root_path='/', title=None, user_config_file=None, read_only=False):
    '''
Run a server for topic model viewer

Parameters
----------
model: tomotopy.LDAModel or its derived class
    A trained topic model instance to be visualized.
host: str
    The host name to bind the server. Default is 'localhost'.
port: int
    The port number to bind the server. Default is 80.
root_path: str
    The root path of the viewer. Default is '/'.
title: str
    The title of the viewer in a web browser. Default is the class name of the model.
user_config_file: str
    The path to a JSON file to store the user configurations. Default is `None`. If None, the user configurations are not saved.
read_only: bool
    If True, the viewer will be read-only, that is the user cannot change topic labels. Default is False.

Note
----
It is not recommended to use it in a production web service, 
because this uses python's built-in `http.server` module which is not designed for high-performance production environments.
    '''
    import tomotopy as tp
    if not isinstance(model, tp.LDAModel):
        raise ValueError(f'`model` must be an instance of tomotopy.LDAModel, but {model!r} was given.')
    
    if title is None:
        title = type(model).__name__ + ' Viewer'
    
    template = _prepare_template()

    with http.server.ThreadingHTTPServer((host, port), ViewerHandler) as httpd:
        httpd.title = title
        httpd.model = model
        httpd.root_path = root_path
        httpd.template = template
        httpd.user_config_file = user_config_file
        httpd.user_config = None
        httpd.read_only = read_only
        httpd.filter = DocumentFilter(model)
        print(f'Serving a topic model viewer at http://{httpd.server_address[0]}:{httpd.server_address[1]}/')
        httpd.serve_forever()
