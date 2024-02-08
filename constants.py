stop_words_to_include = ["must", "no"]
aspect_mapping = {'NULL' : 0,
                'RESTAURANT': 1,
                'FOOD': 2,
                'DRINKS': 3,
                'AMBIENCE': 4,
                'SERVICE': 5,
                'LOCATION': 6}

opinion_mapping = {'NULL' : 0,
                'GENERAL': 1,
                'PRICES': 2,
                'QUALITY': 3,
                'STYLE_OPTIONS': 4,
                'MISCELLANEOUS': 5}

map_to_aspect = {0: 'NULL',
                  1: 'RESTAURANT',
                  2: 'FOOD',
                  3: 'DRINKS',
                  4: 'AMBIENCE',
                  5: 'SERVICE',
                  6: 'LOCATION'}

map_to_opinion = {0: 'NULL',
                   1: 'GENERAL',
                   2: 'PRICES',
                   3: 'QUALITY',
                   4: 'STYLE_OPTIONS',
                   5: 'MISCELLANEOUS'}