import random
import numpy as np

a = """Madison, Wisconsin; 269,196
Fort Wayne, Indiana; 265,974
Des Moines, Iowa; 212,031
Aurora, Illinois; 179,266
Grand Rapids, Michigan; 197,416
Overland Park, Kansas; 197,106
Akron, Ohio; 189,347
Sioux Falls, South Dakota; 196,528
Springfield, Missouri; 169,724
Kansas City, Kansas; 508,394
Rockford, Illinois; 147,711
Joliet, Illinois; 150,371
Naperville, Illinois; 149,104
Dayton, Ohio; 137,571
Warren, Michigan; 138,130
Olathe, Kansas; 143,014
Sterling Heights, Michigan; 131,996
Cedar Rapids, Iowa; 130,330
Topeka, Kansas; 127,139
Fargo, North Dakota; 125,804
Rochester, Minnesota; 124,599
Evansville, Indiana; 119,806
Ann Arbor, Michigan; 119,303
Columbia, Missouri; 118,620
Independence, Missouri; 117,369
Springfield, Illinois; 116,313
Peoria, Illinois; 115,424
Lansing, Michigan; 115,222
Elgin, Illinois; 112,628
Green Bay, Wisconsin; 104,796"""

ab = a.split('\n')
ab = [(d.split(';')[0].split(',')[0], int(d.split(';')[1].replace(',', '').replace(' ', ''))) for d in ab]
print(ab)