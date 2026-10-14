"""Check the offline review's executed browser DOM, without image synthesis."""
import json
from html.parser import HTMLParser
from pathlib import Path


class ReviewDOM(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_metrics = False
        self.metrics = ''
        self.images = {}
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag=='pre' and attrs.get('id')=='metrics':
            self.in_metrics = True
        if tag=='img':
            self.images[attrs.get('id')] = attrs.get('src')
    def handle_endtag(self, tag):
        if tag=='pre':
            self.in_metrics = False
    def handle_data(self, data):
        if self.in_metrics:
            self.metrics += data


parser = ReviewDOM()
parser.feed(Path('outputs/badminton/browser_dom.html').read_text(encoding='utf-8-sig'))
metrics = json.loads(parser.metrics)
assert metrics['summary']['accepted']>0
assert parser.images.get('cam1','').endswith('_cam1.jpg')
assert parser.images.get('cam2','').endswith('_cam2.jpg')
print(json.dumps({'rendered':True,'images':parser.images,'summary':metrics['summary']},indent=2))
