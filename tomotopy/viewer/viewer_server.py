import os
import re
import io
import urllib.parse
import html
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

def hue2rgb(h:float):
    h = h % 6.0
    if h < 1:
        return (255, int(255 * h), 0)
    elif h < 2:
        return (int(255 * (2 - h)), 255, 0)
    elif h < 3:
        return (0, 255, int(255 * (h - 2)))
    elif h < 4:
        return (0, int(255 * (4 - h)), 255)
    elif h < 5:
        return (int(255 * (h - 4)), 0, 255)
    else:
        return (255, 0, int(255 * (6 - h)))

topic_colors = [hue2rgb((i * 7) / 40 * 6) for i in range(40)]
topic_styles = [(i * 7 // 40) % 3 for i in range(40)]

class DocumentFilter:
    def __init__(self, model, max_cache_size=5) -> None:
        self.model = model
        self.max_cache_size = max_cache_size
        self._cached = {}
        self._cached_keys = []
    
    def _sort_and_filter(self, sort_key:int, filter_target:int, filter_value:float):
        results = []
        for i, doc in enumerate(self.model.docs):
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

    def _get_cached_filter_result(self, sort_key:int, filter_target:int, filter_value:float):
        if sort_key < 0 and filter_value <= 0:
            # return None for no filtering nor sorting
            return None
        key = (sort_key, filter_target, filter_value)
        if key in self._cached:
            return self._cached[key]
        else:
            result = self._sort_and_filter(sort_key, filter_target, filter_value)
            if len(self._cached_keys) >= self.max_cache_size:
                del self._cached[self._cached_keys.pop(0)]
            self._cached[key] = result
            self._cached_keys.append(key)
            return result

    def get(self, sort_key:int, filter_target:int, filter_value:float, index:slice):
        # return (doc_indices, total_docs_filtered)
        result = self._get_cached_filter_result(sort_key, filter_target, filter_value)
        if result is None:
            return list(range(index.start, min(index.stop, len(self.model.docs)))), len(self.model.docs)
        else:
            return result[index], len(result)
    

class ViewerHandler(http.server.SimpleHTTPRequestHandler):

    handlers = [
        (r'/?', 'overview'),
        (r'/document/?', 'document'),
        (r'/topic/?', 'topic'),
        (r'/api/document/(\d+)', 'api_document'),
    ]

    num_docs_per_page = 30

    @property
    def title(self):
        return self.server.title
    
    @property
    def model(self):
        return self.server.model
    
    @property
    def tomotopy_version(self):
        import tomotopy
        return tomotopy.__version__

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
        for pattern, handler in self.handlers:
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
        page = int(self.arguments.get('p', '0'))
        
        doc_indices, filtered_docs = self.server.filter.get(
            sort_key, 
            filter_target, 
            filter_value / 100, 
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
                    filtered_docs=filtered_docs if filter_value > 0 else None,
                    total_docs=total_docs,
                    documents=documents,
                    sort_key=sort_key,
                    filter_target=filter_target,
                    filter_value=filter_value,
                    )
    
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
                topic_words = [(self.model.vocabs[w], topic_word_dist[k, w]) for w in top_word]
                topics.append(Topic(k, topic_words))
        else:
            for k in range(self.model.k):
                topic_words = self.model.get_topic_words(k, top_n)
                max_dist = max(max_dist, topic_words[0][1])
                topics.append(Topic(k, topic_words))
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.render(action='topic',
                    topics=topics,
                    max_dist=max_dist,
                    top_n=top_n,)

    def get_api_document(self, doc_id):
        doc_id = int(doc_id)
        doc = self.model.docs[doc_id]
        chunks = []
        raw = doc.raw
        last = 0
        for (s, e), topic, w in zip(doc.span, doc.topics, doc.words):
            if topic < 0: continue
            if last < s:
                chunks.append(html.escape(raw[last:s]))
            chunks.append(f'<span class="topic-color-{topic % 40}-040 topic-action" data-topic-id="{topic}" data-bs-toggle="tooltip" data-bs-title="Topic #{topic}: {self.model.vocabs[w]}">{html.escape(raw[s:e])}</span>')
            last = e
        if last < len(raw):
            chunks.append(html.escape(raw[last:]))
        html_cont = '<p>' + ''.join(chunks).strip().replace('\n', '<br>') + '</p>'
        
        chunks = ['<table class="table table-striped table-hover topic-dist">']
        for topic_id, dist in doc.get_topics(top_n=-1):
            chunks.append(
f'''<tr class="topic-action" data-topic-id="{topic_id}">
    <th scope="row"><a href="javascript:void(0);">Topic # {topic_id}</a></th>
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

def open_viewer(model, host='localhost', port=80, title=None):
    import tomotopy as tp
    if not isinstance(model, tp.LDAModel):
        raise ValueError(f'`model` must be an instance of tomotopy.LDAModel, but {model!r} was given.')
    
    if title is None:
        title = type(model).__name__ + ' Viewer'
    
    template = _prepare_template()

    with http.server.ThreadingHTTPServer((host, port), ViewerHandler) as httpd:
        httpd.title = title
        httpd.model = model
        httpd.template = template
        httpd.filter = DocumentFilter(model)
        print(f'Serving a topic model viewer at http://{httpd.server_address[0]}:{httpd.server_address[1]}/')
        httpd.serve_forever()

