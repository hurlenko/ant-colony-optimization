from threading import Thread
import numpy as np


def gen_matrix(n, a=5, b=1000):
    return np.random.randint(a, b, (n, n))

class ACO:
    class Ant(Thread):
        def __init__(self,
                     init_location,
                     possible_locations,
                     pheromone_map,
                     distance_callback,
                     alpha,
                     beta,
                     first_pass=False):
            Thread.__init__(self)

            self.init_location = init_location
            self.possible_locations = possible_locations
            self.route = []
            self.distance_traveled = 0.0
            self.location = init_location
            self.pheromone_map = pheromone_map
            self.distance_callback = distance_callback
            self.alpha = alpha
            self.beta = beta
            self.first_pass = first_pass
            self.update_route(init_location)

            # self.possible_locations.append(init_location)
            self.tour_complete = False

        def run(self):
            while self.possible_locations:
                nxt = self.pick_path()
                self.traverse(self.location, nxt)
            self.possible_locations.append(self.init_location)
            self.traverse(self.location, self.init_location)
            self.tour_complete = True

        def pick_path(self):
            if self.first_pass:
                import random
                rnd = random.choice(self.possible_locations)
                while rnd == self.init_location and len(self.possible_locations) > 1:
                    rnd = random.choice(self.possible_locations)
                return rnd

            attractiveness = dict()
            sum_total = 0.0
            for possible_next_location in self.possible_locations:
                pheromone_amount = float(self.pheromone_map[self.location][possible_next_location])
                distance = float(self.distance_callback(self.location, possible_next_location))
                attractiveness[possible_next_location] = pow(pheromone_amount, self.alpha) * pow(1 / distance,
                                                                                                 self.beta)
                sum_total += attractiveness[possible_next_location]
                # attractiveness[self.init_location] = 0
            if sum_total == 0.0:
                def next_up(x):
                    import math
                    import struct
                    if math.isnan(x) or (math.isinf(x) and x > 0):
                        return x
                    if x == 0.0:
                        x = 0.0
                    n = struct.unpack('<q', struct.pack('<d', x))[0]
                    if n >= 0:
                        n += 1
                    else:
                        n -= 1
                    return struct.unpack('<d', struct.pack('<q', n))[0]

                for key in attractiveness:
                    attractiveness[key] = next_up(attractiveness[key])
                sum_total = next_up(sum_total)
            import random
            toss = random.random()

            cummulative = 0
            for possible_next_location in attractiveness:
                weight = (attractiveness[possible_next_location] / sum_total)
                if toss <= weight + cummulative:
                    return possible_next_location
                cummulative += weight
            # return self.init_location

        def traverse(self, start, end):
            self.update_route(end)
            self.update_distance_traveled(start, end)
            self.location = end

        def update_route(self, new):
            self.route.append(new)
            self.possible_locations = list(self.possible_locations)
            self.possible_locations.remove(new)

        def update_distance_traveled(self, start, end):
            self.distance_traveled += float(self.distance_callback(start, end))

        def get_route(self):
            if self.tour_complete:
                return self.route
            return None

        def get_distance_traveled(self):
            if self.tour_complete:
                return self.distance_traveled
            return None

    def __init__(self,
                 nodes_num,
                 distance_matrix,
                 start,
                 ant_count,
                 alpha,
                 beta,
                 pheromone_evaporation_coefficient,
                 pheromone_constant,
                 iterations):
        self.nodes = list(range(nodes_num))
        self.nodes_num = nodes_num
        self.distance_matrix = distance_matrix
        self.pheromone_map = self.init_matrix(nodes_num)
        self.ant_updated_pheromone_map = self.init_matrix(nodes_num)
        self.start = 0 if start is None else start
        self.ant_count = ant_count
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pheromone_evaporation_coefficient = float(pheromone_evaporation_coefficient)
        self.pheromone_constant = float(pheromone_constant)
        self.iterations = iterations
        self.first_pass = True
        self.ants = self._init_ants(self.start)
        self.shortest_distance = None
        self.shortest_path_seen = None

    def get_distance(self, start, end):
        return self.distance_matrix[start][end]

    def init_matrix(self, size, value=0.0):
        ret = []
        for row in range(size):
            ret.append([float(value) for _ in range(size)])
        return ret

    def _init_ants(self, start):
        if self.first_pass:
            return [self.Ant(start, self.nodes, self.pheromone_map, self.get_distance,
                             self.alpha, self.beta, first_pass=True) for _ in range(self.ant_count)]
        for ant in self.ants:
            ant.__init__(start, self.nodes, self.pheromone_map, self.get_distance, self.alpha, self.beta)

    def update_pheromone_map(self):
        for start in range(len(self.pheromone_map)):
            for end in range(len(self.pheromone_map)):
                # tau_xy <- (1-rho)*tau_xy	(ACO)
                self.pheromone_map[start][end] *= (1 - self.pheromone_evaporation_coefficient)
                # tau_xy <- tau_xy + delta tau_xy_k
                # delta tau_xy_k = Q / L_k
                self.pheromone_map[start][end] += self.ant_updated_pheromone_map[start][end]

    def populate_ant_updated_pheromone_map(self, ant):
        route = ant.get_route()
        for i in range(len(route) - 1):
            current_pheromone_value = float(self.ant_updated_pheromone_map[route[i]][route[i + 1]])
            # delta tau_xy_k = Q / L_k
            new_pheromone_value = self.pheromone_constant / ant.get_distance_traveled()

            self.ant_updated_pheromone_map[route[i]][route[i + 1]] = current_pheromone_value + new_pheromone_value
            self.ant_updated_pheromone_map[route[i + 1]][route[i]] = current_pheromone_value + new_pheromone_value

    def mainloop(self):
        for it in range(self.iterations):
            for ant in self.ants:
                ant.start()
            for ant in self.ants:
                ant.join()

            for ant in self.ants:
                self.populate_ant_updated_pheromone_map(ant)
                if not self.shortest_distance:
                    self.shortest_distance = ant.get_distance_traveled()

                if not self.shortest_path_seen:
                    self.shortest_path_seen = ant.get_route()

                if ant.get_distance_traveled() < self.shortest_distance:
                    self.shortest_distance = ant.get_distance_traveled()
                    self.shortest_path_seen = ant.get_route()

            self.update_pheromone_map()

            if self.first_pass:
                self.first_pass = False

            self._init_ants(self.start)

            self.ant_updated_pheromone_map = self.init_matrix(self.nodes_num, value=0)
            if (it + 1) % 50 == 0:
                print('{0}/{1} Searching...'.format(it + 1, self.iterations))

        ret = []
        for ids in self.shortest_path_seen:
            ret.append(self.nodes[ids])

        return ret


START = None
ANT_COUNT = 100
ALPHA = 1.1
BETA = 1.1
PHER_EVAP_COEFF = .70
PHER_CONSTANT = 3000.0
ITERATIONS = 400


def main():
    # matrix = gen_matrix(40)
    # np.savetxt('distance_matrix.txt', matrix, fmt='%g', delimiter=' ')
    distance_matrix = np.loadtxt(open('distance_matrix.txt', 'rb'), delimiter=' ')

    colony = ACO(len(distance_matrix[0]),
                 distance_matrix,
                 START,
                 ANT_COUNT,
                 ALPHA,
                 BETA,
                 PHER_EVAP_COEFF,
                 PHER_CONSTANT,
                 ITERATIONS)
    answer = colony.mainloop()
    print(np.array(answer) + 1)
    print(colony.shortest_distance)

if __name__ == '__main__':
    main()
