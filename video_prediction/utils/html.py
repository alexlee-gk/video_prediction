import os

import dominate
from dominate.tags import *


class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))
        self.t = None

    def get_image_dir(self):
        return self.img_dir

    def add_header1(self, str):
        with self.doc:
            h1(str)

    def add_header2(self, str):
        with self.doc:
            h2(str)

    def add_header3(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_row(self, txts, colspans=None):
        if self.t is None:
            self.add_table()
        with self.t:
            with tr():
                if colspans:
                    assert len(txts) == len(colspans)
                    colspans = [dict(colspan=str(colspan)) for colspan in colspans]
                else:
                    colspans = [dict()] * len(txts)
                for txt, colspan in zip(txts, colspans):
                    style = "word-break: break-all;" if len(str(txt)) > 80 else "word-wrap: break-word;"
                    with td(style=style, halign="center", valign="top", **colspan):
                        with p():
                            if txt is not None:
                                p(txt)

    def add_images(self, ims, txts, links, colspans=None, height=None, width=400):
        image_style = ''
        if height is not None:
            image_style += "height:%dpx;" % height
        if width is not None:
            image_style += "width:%dpx;" % width
        if self.t is None:
            self.add_table()
        with self.t:
            with tr():
                if colspans:
                    assert len(txts) == len(colspans)
                    colspans = [dict(colspan=str(colspan)) for colspan in colspans]
                else:
                    colspans = [dict()] * len(txts)
                for im, txt, link, colspan in zip(ims, txts, links, colspans):
                    with td(style="word-wrap: break-word;", halign="center", valign="top", **colspan):
                        with p():
                            if im is not None and link is not None:
                                with a(href=os.path.join('images', link)):
                                    img(style=image_style, src=os.path.join('images', im))
                            if im is not None and link is not None and txt is not None:
                                br()
                            if txt is not None:
                                p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.jpg' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.jpg' % n)
    html.add_images(ims, txts, links)
    html.save()
