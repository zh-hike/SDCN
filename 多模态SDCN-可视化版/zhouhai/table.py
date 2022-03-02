
class MyhtmlTable:

    def __init__(self, shape, data, table_title):
        """
        :param shape: 二元组，（行数，列数）
        :param data: ndarray
        """
        self.row_num = shape[0]
        self.col_num = shape[1]
        print("创建我的表格：row_num=" + str(self.row_num) + " col_num=" + str(self.col_num))
        self.data = data

        # 生成style
        self.style = '<style>\n'
        self.style += '    .lastcol {\n'
        self.style += '        width:160px;\n'
        self.style += '    }\n'
        self.style += '    th,td {\n'
        self.style += '        font-size:21px;\n'
        self.style += '        text-align:center;\n'
        self.style += '    }\n'
        self.style += '    .title {\n'
        self.style += '        font-size:26px;\n'
        self.style += '    }\n'
        self.style += '    table {\n'
        self.style += '        margin:auto;\n'
        self.style += '    }\n'
        self.style += '</style>\n'

        # 生成table
        self.title = '<p class="title" align="center">' + table_title + '</p>\n'
        self.table = self.style + self.title + '<table border="2" cellspacing="0">\n'
        self.table += self.__set_table_head(data[0])
        for i in range(self.row_num-1):
            self.table += self.__add_row(data[i + 1])
        self.table += "</table>"

    def __set_table_head(self, row_data):
        row_line = "    <thead><tr>\n"
        for i in range(self.col_num):
            if i == self.col_num - 1:
                row_line += '        <th class="lastcol">' + str(row_data[i]) + "</th>\n"
            else:
                row_line += "        <th>" + str(row_data[i]) + "</th>\n"
        row_line += "    </tr></thead>\n"
        return row_line

    def __add_row(self, row_data):
        row_line = "    <tr>\n"
        for i in range(self.col_num):
            if i == self.col_num - 1:
                row_line += '        <td class="lastcol">' + str(row_data[i]) + "</td>\n"
            else:
                row_line += "        <td>" + str(row_data[i]) + "</td>\n"

        row_line += "    </tr>\n"
        return row_line

    def print(self):
        print(self.table)

    def get_table(self):
        return self.table

