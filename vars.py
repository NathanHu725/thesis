from connections import *
import numpy as np
from numpy.random import binomial

class VarGetter:
    def get_start_nodes(self):
        return self.start_nodes

    def get_dvars(self):
        return self.dvars

    def get_time_vars(self):
        return self.time_vars

    def get_travel_vars(self):
        return self.travel_vars_rad

    def get_cities(self):
        return self.cities_extended

    def get_distances(self):
        return self.distances_extended

    def get_cities_small(self):
        return self.cities_small

    def get_distances_small(self):
        return self.distances_small

    def constant_threshold(self, infected, total_pop):
        threshold_pop = 400000
        tested_positive = self.num_to_test_positive(infected)
        return tested_positive > threshold_pop

    def percent_threshold(self, infected, total_pop):
        threshold = self.threshold
        tested_positive = self.num_to_test_positive(infected)
        return threshold * total_pop < tested_positive

    def num_to_test_positive(self, testers):
        return binomial(max(testers, 1), self.testing_vars['positive_to_positive'])

    def constant_beta(self, time, beta):
        return beta

    def sin_beta(self, time, beta):
        # This variable indicates how many times above and below the beta value we go, should be less than one to stay positive
        return beta * (1 + self.spike * np.sin(time * 2 * np.pi / 365))

    def __init__(self):
        self.threshold = 1
        self.start_nodes = ['Chicago']
        self.spike = .75

        self.cities_extended = [('Milwaukee', 570000), ('Rockford', 147000), ('Gary', 70000), ('Chicago', 2670000), ('Minneapolis', 425000), ('St. Paul', 307000), ('Madison', 270000), ('Indianapolis', 882000), ('Fort Wayne', 265974), ('Des Moines', 212031), ('Aurora', 179266), ('Grand Rapids', 197416), ('Overland Park', 197106), ('Akron', 189347), ('Sioux Falls', 196528), ('Springfield MO', 169724), ('Kansas City MO', 508394), ('Joliet', 150371), ('Naperville', 149104), ('Dayton', 137571), ('Warren', 138130), ('Olathe', 143014), ('Sterling Heights', 131996), ('Cedar Rapids', 130330), ('Topeka', 127139), ('Fargo', 125804), ('Rochester', 124599), ('Evansville', 119806), ('Ann Arbor', 119303), ('Columbia', 118620), ('Independence', 117369), ('Springfield IL', 116313), ('Peoria', 115424), ('Lansing', 115222), ('Elgin', 112628), ('Green Bay', 104796), ('Toledo', 268508), ('Lincoln', 293000), ('St. Louis', 301000), ('Columbus', 905000), ('Detroit', 639000), ('Kansas City KS', 477000), ('Omaha', 463000), ('Wichita', 397000), ('Cleveland', 373000), ('Cincinnati', 309000)]
        self.cities = [('Milwaukee', 570000), ('Rockford', 147000), ('Gary', 70000), ('Chicago', 2670000), ('Minneapolis', 425000), ('St. Paul', 307000), ('Madison', 270000), ('Indianapolis', 882000)]
        self.cities_one = [('Chicago', 2670000)]
        self.cities_small = [('Chicago', 2670000), ('Milwaukee', 570000), ('Rockford', 147000), ('Gary', 70000)]

        self.distances_one = [[0]]

        self.distances_small = [[0, 100, float('inf'), float('inf')], 
                    [100, 0, float('inf'), 50], 
                    [float('inf'), float('inf'), 0, 30],
                    [float('inf'), 50, 30, 0]]

        self.distances = [[0, 100, 133, 83, 406, 329, 80, 280], 
                    [100, 0, 125, 90, 335, 327, 73, 271], 
                    [133, 125, 0, 30, 437, 429, 187, 151],
                    [83, 90, 30, 0, 408, 401, 147, 183],
                    [406, 335, 437, 408, 0, 12, 268, 591],
                    [329, 327, 429, 401, 12, 0, 376, 42],
                    [80, 73, 187, 147, 268, 376, 0, 330],
                    [280, 271, 151, 183, 591, 42, 330, 0]]

        self.distances_extended = [
            [0, 100, 133, 83, 406, 329, 80, 280, 253, 373, 113, 269, 577, 461, 502, 587, 565, 123, 103, 394, 381, 587, 386, 244, 627, 570, 276, 385, 331, 461, 567, 277, 222, 308, 101, 116, 337, 562, 372, 452, 372, 565, 507, 760, 438, 392], 
            [100, 0, 125, 90, 335, 327, 73, 271, 249, 287, 72, 270, 490, 457, 500, 509, 481, 109, 82, 386, 383, 501, 387, 164, 540, 568, 274, 363, 332, 372, 481, 199, 135, 309, 52, 202, 333, 475, 293, 445, 374, 479, 420, 673, 434, 384], 
            [133, 125, 0, 30, 437, 429, 187, 151, 132, 350, 70, 148, 526, 340, 603, 516, 514, 43, 57, 266, 261, 536, 265, 262, 576, 671, 377, 274, 211, 390, 517, 206, 170, 188, 69, 235, 216, 538, 301, 324, 252, 517, 483, 709, 317, 264],
            [83, 90, 30, 0, 408, 401, 147, 183, 162, 333, 41, 179, 521, 370, 574, 511, 510, 45, 32, 298, 291, 532, 295, 245, 572, 642, 348, 292, 241, 385, 512, 201, 166, 218, 40, 206, 247, 521, 296, 356, 282, 512, 466, 704, 347, 296],
            [406, 335, 437, 408, 0, 12, 268, 591, 569, 244, 398, 590, 447, 777, 236, 599, 436, 429, 409, 706, 702, 458, 706, 276, 497, 235, 86, 689, 652, 468, 438, 525, 453, 629, 371, 277, 653, 432, 559, 764, 693, 438, 377, 630, 754, 704],
            [329, 327, 429, 401, 12, 0, 376, 42, 559, 248, 389, 580, 452, 767, 245, 604, 440, 419, 399, 696, 692, 462, 697, 244, 502, 246, 77, 680, 642, 472, 442, 515, 422, 619, 361, 268, 643, 437, 528, 754, 683, 442, 382, 635, 744, 694],
            [80, 73, 187, 147, 268, 376, 0, 330, 308, 293, 137, 328, 496, 516, 443, 574, 484, 168, 148, 445, 441, 506, 445, 165, 546, 511, 217, 428, 391, 412, 487, 264, 201, 368, 110, 144, 392, 481, 359, 503, 432, 487, 426, 679, 493, 443],
            [280, 271, 151, 183, 591, 42, 330, 0, 124, 477, 216, 262, 493, 299, 742, 457, 481, 190, 204, 117, 303, 502, 307, 389, 543, 828, 535, 171, 265, 358, 473, 212, 212, 254, 219, 393, 225, 643, 242, 175, 286, 484, 610, 675, 316, 112],
            [253, 249, 132, 162, 569, 559, 308, 124, 0, 479, 199, 175, 622, 211, 734, 587, 611, 172, 186, 126, 179, 632, 183, 391, 673, 802, 508, 297, 142, 488, 603, 291, 263, 131, 200, 366, 102, 667, 371, 158, 162, 614, 612, 804, 204, 179],
            [373, 287, 350, 333, 244, 248, 293, 477, 479, 0, 298, 495, 204, 688, 282, 356, 193, 309, 308, 590, 608, 215, 612, 127, 255, 478, 211, 504, 557, 225, 195, 337, 264, 535, 305, 434, 565, 189, 339, 648, 599, 195, 134, 387, 665, 585],
            [113, 72, 70, 41, 398, 389, 137, 216, 199, 298, 0, 215, 501, 408, 564, 484, 489, 22, 10, 331, 327, 511, 332, 210, 551, 632, 338, 307, 277, 358, 492, 174, 120, 254, 21, 227, 284, 486, 269, 389, 318, 492, 431, 684, 385, 329],
            [269, 270, 148, 179, 590, 580, 328, 262, 175, 495, 215, 0, 671, 323, 759, 661, 660, 189, 203, 303, 155, 682, 165, 408, 721, 827, 152, 432, 132, 535, 662, 351, 315, 68, 218, 383, 186, 684, 446, 321, 157, 662, 629, 854, 300, 356],
            [577, 490, 526, 521, 447, 452, 496, 493, 622, 204, 501, 671, 0, 794, 371, 164, 11, 483, 497, 612, 782, 13, 787, 331, 65, 609, 415, 422, 732, 136, 20, 320, 392, 709, 509, 637, 720, 203, 258, 670, 773, 13, 193, 185, 811, 599], 
            [461, 457, 340, 370, 777, 767, 516, 299, 211, 688, 408, 323, 794, 0, 940, 758, 782, 378, 393, 195, 209, 803, 213, 597, 844, 1008, 714, 453, 190, 659, 774, 506, 505, 256, 406, 572, 137, 873, 543, 125, 192, 785, 818, 976, 39, 232],
            [502, 500, 603, 574, 236, 245, 443, 742, 734, 282, 564, 759, 371, 940, 0, 525, 360, 574, 574, 855, 867, 375, 872, 352, 344, 242, 237, 770, 817, 485, 370, 597, 529, 794, 536, 492, 818, 212, 607, 913, 858, 356, 181, 482, 919, 850],
            [587, 509, 516, 511, 599, 604, 574, 457, 587, 356, 484, 661, 164, 758, 525, 0, 166, 473, 487, 575, 761, 171, 766, 405, 222, 767, 567, 379, 722, 167, 169, 310, 382, 699, 521, 701, 684, 360, 215, 633, 744, 169, 350, 247, 775, 563],
            [565, 481, 514, 510, 436, 440, 484, 481, 611, 193, 489, 660, 11, 782, 360, 166, 0, 472, 486, 601, 771, 22, 775, 320, 63, 602, 404, 411, 721, 125, 10, 309, 381, 698, 497, 626, 709, 195, 247, 659, 762, 3, 185, 194, 800, 588],
            [123, 109, 43, 45, 429, 419, 168, 190, 172, 309, 22, 189, 483, 378, 574, 473, 472, 0, 19, 305, 301, 495, 305, 221, 534, 665, 371, 296, 251, 348, 475, 164, 128, 228, 57, 238, 258, 497, 259, 363, 292, 475, 441, 667, 359, 303],
            [103, 82, 57, 32, 409, 399, 148, 204, 186, 308, 10, 203, 497, 393, 574, 487, 486, 19, 0, 319, 315, 509, 320, 220, 549, 642, 348, 306, 265, 362, 489, 179, 143, 242, 25, 219, 272, 496, 274, 377, 306, 489, 441, 682, 373, 317],
            [394, 386, 266, 298, 706, 696, 445, 117, 126, 590, 331, 303, 612, 195, 855, 575, 601, 305, 319, 0, 226, 621, 230, 501, 663, 943, 649, 273, 196, 477, 592, 324, 325, 262, 333, 508, 148, 755, 361, 71, 209, 604, 722, 794, 212, 54],
            [381, 383, 261, 291, 702, 692, 441, 303, 179, 608, 327, 155, 782, 209, 867, 761, 771, 301, 315, 226, 0, 794, 5, 520, 834, 940, 646, 475, 52, 648, 775, 464, 428, 89, 330, 496, 88, 796, 559, 219, 19, 775, 741, 967, 187, 280],
            [587, 501, 536, 532, 458, 462, 506, 502, 632, 215, 511, 682, 13, 803, 375, 171, 22, 495, 509, 621, 794, 0, 797, 342, 56, 616, 426, 431, 743, 145, 33, 331, 403, 720, 519, 648, 730, 209, 268, 679, 784, 21, 199, 174, 820, 609],
            [386, 387, 265, 295, 706, 697, 445, 307, 183, 612, 332, 165, 787, 213, 872, 766, 775, 305, 320, 230, 5, 797, 0, 524, 838, 944, 650, 479, 55, 651, 778, 468, 432, 93, 334, 499, 79, 800, 562, 223, 23, 778, 745, 971, 190, 284],
            [244, 164, 262, 245, 276, 244, 165, 389, 391, 127, 210, 408, 331, 597, 352, 405, 320, 221, 220, 501, 520, 342, 524, 0, 381, 509, 170, 448, 471, 247, 321, 250, 177, 448, 218, 305, 478, 316, 283, 561, 512, 321, 260, 514, 578, 498],
            [627, 540, 576, 572, 497, 502, 546, 543, 673, 255, 551, 721, 65, 844, 344, 222, 63, 534, 549, 663, 834, 56, 838, 381, 0, 583, 466, 473, 783, 187, 72, 371, 443, 760, 559, 688, 771, 167, 309, 721, 824, 61, 163, 138, 862, 650],
            [570, 568, 671, 642, 235, 246, 511, 828, 802, 478, 632, 827, 609, 1008, 242, 767, 602, 665, 642, 943, 940, 616, 944, 509, 583, 0, 321, 924, 887, 724, 609, 760, 687, 864, 606, 512, 888, 475, 793, 999, 928, 595, 420, 721, 989, 939],
            [276, 274, 377, 348, 86, 77, 217, 535, 508, 211, 338, 152, 415, 714, 237, 567, 404, 371, 348, 649, 646, 426, 650, 170, 466, 321, 0, 630, 592, 416, 405, 419, 346, 569, 311, 267, 593, 400, 452, 704, 633, 405, 344, 598, 694, 644],
            [385, 363, 274, 292, 689, 680, 428, 171, 297, 504, 307, 432, 422, 453, 770, 379, 411, 296, 306, 273, 475, 431, 479, 448, 473, 924, 630, 0, 436, 289, 404, 230, 269, 425, 325, 500, 396, 607, 166, 328, 457, 415, 596, 605, 470, 221],
            [331, 332, 211, 241, 652, 642, 391, 265, 142, 557, 277, 132, 732, 190, 817, 722, 721, 251, 265, 196, 52, 743, 55, 471, 783, 887, 592, 436, 0, 597, 724, 414, 378, 65, 280, 445, 53, 746, 508, 188, 42, 724, 691, 916, 168, 249],
            [461, 372, 390, 385, 468, 472, 412, 358, 488, 225, 358, 535, 136, 659, 485, 167, 125, 348, 362, 477, 648, 145, 651, 247, 187, 724, 416, 289, 597, 0, 117, 185, 256, 573, 395, 553, 586, 320, 125, 536, 646, 128, 310, 318, 677, 465],
            [567, 481, 517, 512, 438, 442, 487, 473, 603, 195, 492, 662, 20, 774, 370, 169, 10, 475, 489, 592, 775, 33, 778, 321, 72, 609, 405, 404, 724, 117, 0, 311, 383, 700, 499, 628, 702, 205, 240, 651, 764, 13, 195, 206, 792, 581],
            [277, 199, 206, 201, 525, 515, 264, 212, 291, 337, 174, 351, 320, 506, 597, 310, 309, 164, 179, 324, 464, 331, 468, 250, 371, 760, 419, 230, 414, 185, 311, 0, 74, 391, 213, 393, 421, 432, 96, 384, 455, 311, 422, 504, 522, 321],
            [222, 135, 170, 166, 453, 422, 201, 212, 263, 264, 120, 315, 392, 505, 529, 382, 381, 128, 143, 325, 428, 403, 432, 177, 443, 687, 346, 269, 378, 256, 383, 74, 0, 355, 153, 329, 385, 453, 168, 384, 419, 384, 398, 576, 486, 321],
            [308, 309, 188, 218, 629, 619, 368, 254, 131, 535, 254, 68, 709, 256, 794, 699, 698, 228, 242, 262, 89, 720, 93, 448, 760, 864, 569, 425, 65, 573, 700, 391, 355, 0, 257, 423, 119, 723, 486, 254, 90, 701, 668, 894, 233, 314],
            [101, 52, 69, 40, 371, 361, 110, 219, 200, 305, 21, 218, 509, 406, 536, 521, 497, 57, 25, 333, 330, 519, 334, 218, 559, 606, 311, 325, 280, 395, 499, 213, 153, 257, 0, 215, 286, 493, 307, 392, 321, 499, 438, 691, 386, 331],
            [116, 202, 235, 206, 277, 268, 144, 393, 366, 434, 227, 383, 637, 572, 492, 701, 626, 238, 219, 508, 496, 648, 499, 305, 688, 512, 267, 500, 445, 553, 628, 393, 329, 423, 215, 0, 451, 622, 487, 567, 487, 628, 567, 820, 552, 506],
            [337, 333, 216, 247, 653, 643, 392, 225, 102, 565, 284, 186, 720, 137, 818, 684, 709, 258, 272, 148, 88, 730, 79, 478, 771, 888, 593, 396, 53, 586, 702, 421, 385, 119, 286, 451, 0, 750, 473, 141, 58, 715, 695, 905, 115, 202],
            [562, 475, 538, 521, 432, 437, 481, 643, 667, 189, 486, 684, 203, 873, 212, 360, 195, 497, 496, 755, 796, 209, 800, 316, 167, 475, 400, 607, 746, 320, 205, 432, 453, 723, 493, 622, 750, 0, 449, 837, 788, 198, 59, 277, 854, 774],
            [372, 293, 301, 296, 559, 528, 359, 242, 371, 339, 269, 446, 258, 543, 607, 125, 247, 259, 274, 361, 559, 268, 562, 283, 309, 793, 452, 166, 508, 125, 240, 96, 168, 486, 307, 487, 473, 449, 0, 420, 530, 250, 432, 441, 561, 349],
            [452, 445, 324, 356, 764, 754, 503, 175, 158, 648, 389, 321, 670, 125, 913, 633, 659, 363, 377, 71, 219, 679, 223, 561, 721, 999, 704, 328, 188, 536, 651, 384, 384, 254, 392, 567, 141, 837, 420, 0, 202, 662, 781, 853, 142, 106],
            [372, 374, 252, 282, 693, 683, 432, 286, 162, 599, 318, 157, 773, 192, 858, 744, 762, 292, 306, 209, 19, 784, 23, 512, 824, 928, 633, 457, 42, 646, 764, 455, 419, 90, 321, 487, 58, 788, 530, 202, 0, 766, 732, 958, 169, 263],
            [565, 479, 517, 512, 438, 442, 487, 484, 614, 195, 492, 662, 13, 785, 356, 169, 3, 475, 489, 604, 775, 21, 778, 321, 61, 595, 405, 415, 724, 128, 13, 311, 384, 701, 499, 628, 715, 198, 250, 662, 766, 0, 181, 196, 803, 592],
            [507, 420, 483, 466, 377, 382, 426, 610, 612, 134, 431, 629, 193, 818, 181, 350, 185, 441, 441, 722, 741, 199, 745, 260, 163, 420, 344, 596, 691, 310, 195, 422, 398, 668, 438, 567, 695, 59, 432, 781, 732, 181, 0, 300, 800, 719],
            [760, 673, 709, 704, 630, 635, 679, 675, 804, 387, 684, 854, 185, 976, 482, 247, 194, 667, 682, 794, 967, 174, 971, 514, 138, 721, 598, 605, 916, 318, 206, 504, 576, 894, 691, 820, 905, 277, 441, 853, 958, 196, 300, 0, 992, 782],
            [438, 434, 317, 347, 754, 744, 493, 316, 204, 665, 385, 300, 811, 39, 919, 775, 800, 359, 373, 212, 187, 820, 190, 578, 862, 989, 694, 470, 168, 677, 792, 522, 486, 233, 386, 552, 115, 854, 561, 142, 169, 803, 800, 992, 0, 249],
            [392, 384, 264, 296, 704, 694, 443, 112, 179, 585, 329, 356, 599, 232, 850, 563, 588, 303, 317, 54, 280, 609, 284, 498, 650, 939, 644, 221, 249, 465, 581, 321, 321, 314, 331, 506, 202, 774, 349, 106, 263, 592, 719, 782, 249, 0]
        ]

        self.dvars = {
            'beta': 2/5, 
            'beta_fun': self.sin_beta,
            'birth_rate': 1 / (55 * 365),
            'natural_death_rate': 1 / (75 * 365),
            'disease_death_rate': .001,
            'incubation_rate': 1/3,
            'recovery_rate': 1/10,
            'lost_immunity_rate': 1/(365),
            'threshold_function': self.percent_threshold,
            'testing_function': self.num_to_test_positive,
            'quarantine_days': 0,
            'all_quarantine': False
        }

        self.time_vars = {
            'time_step': 2,
            'total_time': 200
        }

        self.testing_vars = {
            'positive_to_positive': .9,
            'negative_to_positive': .02
        }

        self.travel_vars_grav = {
            'A': 0.05,
            'gamma': 2.2,
            'connection': GravityConnection,
            'connection_type': 'Radiation'
        }

        self.travel_vars_rad = {
            'commuter_proportion': .0000288 * 4,
            'adjacency_matrix': self.get_distances(),
            'connection': RadiationConnection,
            'connection_type': 'Radication'
        }
