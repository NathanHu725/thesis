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

[('Milwaukee', 570000), 
('Rockford', 147000),
 ('Gary', 70000),
  ('Chicago', 2670000),
   ('Minneapolis', 425000),
    ('St. Paul', 307000), 
     ('Madison', 269196), 
     ('Indianapolis', 882000), 
     ('Fort Wayne', 265974), 
     ('Des Moines', 212031),
      ('Aurora', 179266), 
      ('Grand Rapids', 197416),
       ('Overland Park', 197106),
        ('Akron', 189347),
         ('Sioux Falls', 196528),
          ('Springfield', 169724),
           ('Kansas City', 508394),
             ('Joliet', 150371), 
             ('Naperville', 149104),
              ('Dayton', 137571),
               ('Warren', 138130),
                ('Olathe', 143014),
                 ('Sterling Heights', 131996),
                  ('Cedar Rapids', 130330),
                   ('Topeka', 127139), 
                   ('Fargo', 125804),
                    ('Rochester', 124599),
                     ('Evansville', 119806),
                      ('Ann Arbor', 119303),
                       ('Columbia', 118620),
                        ('Independence', 117369),
                         ('Springfield', 116313),
                          ('Peoria', 115424),
                           ('Lansing', 115222), 
                           ('Elgin', 112628), 
                           ('Green Bay', 104796), 
                           ('Toledo', 268508),
                            ('Lincoln', 293000),
                             ('St. Louis', 301000),
                              ('Columbus', 905000),
                               ('Detroit', 639000),
                                ('Kansas City', 477000),
                                 ('Omaha', 463000),
                                  ('Wichita', 397000),
                                   ('Cleveland', 373000),
                                    ('Cincinnati', 309000)]