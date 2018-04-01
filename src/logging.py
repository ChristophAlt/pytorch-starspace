from tabulate import tabulate


class TableLogger(object):
    def __init__(self, headers, tablefmt='simple', floatfmt='.4f'):
        self.headers = headers
        self.header_logged = False
        self.tabular_data = []
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        
    def _align_entries(self, entries):
        aligned = [entries.get(h, None) for h in self.headers]
        return aligned
        
    def log(self, **entries):
        tabular_data = self._align_entries(entries)
        self.tabular_data.append(tabular_data)
        table = tabulate(self.tabular_data, headers=self.headers,
            tablefmt=self.tablefmt, floatfmt=self.floatfmt)
        if self.header_logged:
            table = table.rsplit('\n', 2)[-1]
        else:
            self.header_logged = True
        print(table)
