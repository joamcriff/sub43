import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from graphite.data.dataset_utils import load_default_dataset
from graphite.protocol import GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver_multi_2 import NearestNeighbourMultiSolver2
from graphite.utils.graph_utils import get_multi_minmax_tour_distance


@dataclass
class ClusterData:
    depot_index: int
    n_salesmen: int
    distance_matrix: np.ndarray
    coordinates: np.ndarray
    distance_type: str = "Euclidean2D"


@dataclass
class ClusterResult:
    routes: List[List[int]]
    lengths: List[float]


def visualize_routes(node_coords, routes):
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, route in enumerate(routes, start=1):
        # Extract the coordinates for the current route
        route_coords = node_coords[route]

        # Plot the route
        ax.plot(route_coords[:, 0], route_coords[:, 1], marker='o', label=f'Route {i}')

        # Add arrow markers to indicate the direction of travel
        for j in range(len(route) - 1):
            start = route_coords[j]
            end = route_coords[j + 1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='gray', shrinkA=5, shrinkB=5))

    # Add depot marker
    depot_coords = node_coords[0]
    ax.plot(depot_coords[0], depot_coords[1], marker='s', markersize=10, color='red', label='Depot')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Multiple Traveling Salesman Problem Routes')
    ax.legend()
    plt.tight_layout()
    plt.show()

class HGASolver(BaseSolver):
    def __init__(self, problem_types: List[GraphV2Problem] = [GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)
        self.HGA_path = Path("/home/lampham/PycharmProjects/Graphite-Subnet-v2/graphite/m-TSP-julia/test/solve_mtsp.jl")

    async def solve(self, formatted_problem) -> List[List[int]]:
        routes, lengths = self.solve_mtsp(formatted_problem.n_salesmen, formatted_problem.edges, node_coords, formatted_problem.cost_function)
        return routes

    async def solve_cluster(self, cluster_data: ClusterData, julia_path: str) -> ClusterResult:
        """
        Solve a single cluster using Julia HGA solver
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.json"
            output_path = Path(temp_dir) / "output.json"
            print('input_path', input_path)
            print('output_path', output_path)

            # Prepare cluster input
            cluster_input = {
                "n_vehicles": cluster_data.n_salesmen,
                "dist_mtx": cluster_data.distance_matrix.tolist(),
                "coordinates": cluster_data.coordinates.tolist(),
                "distance_type": cluster_data.distance_type
            }

            # Write input file
            with open(input_path, 'w') as f:
                json.dump(cluster_input, f)

            cmd = [
                "julia",
                str(julia_path),
                "--input", str(input_path),
                "--output", str(output_path)
            ]

            import subprocess
            try:
                # Run process and capture output in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,  # This makes output strings instead of bytes
                    bufsize=1  # Line buffered
                )

                # Print output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print("Julia:", output.strip())

                # Get any remaining stderr
                stderr = process.stderr.read()
                if stderr:
                    print("Julia stderr:", stderr)

                # Check return code
                retcode = process.poll()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, cmd)

                # Read results
                with open(output_path, 'r') as f:
                    results = json.load(f)
                print("solve_cluster done")
                return ClusterResult(
                    routes=results["routes"],
                    lengths=results["lengths"]
                )

            except subprocess.CalledProcessError as e:
                print(f"Julia process failed with return code {e.returncode}")
                print(f"Error output: {e.stderr}")
                raise
            except FileNotFoundError:
                raise RuntimeError("Julia solver failed to produce output file")
            except json.JSONDecodeError:
                raise RuntimeError("Invalid JSON output from Julia solver")

    def solve_mtsp(self, n_vehicles, dist_mtx, node_coords, cost_function):
        # Prepare input data
        cluster_data = ClusterData(
            depot_index=0,
            n_salesmen=n_vehicles,
            distance_matrix=np.array(dist_mtx),
            coordinates=np.array(node_coords),
            distance_type=cost_function
        )

        # Solve the cluster
        results = asyncio.run(self.solve_cluster(cluster_data, self.HGA_path))

        if not isinstance(results, ClusterResult):
            raise TypeError("Expected ClusterResult from solve_cluster")

        tours = results.routes
        closed_tours = [[0] + tour + [0] for tour in tours]
        return closed_tours, results.lengths

    def problem_transformations(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
        return problem


import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
    total_start_time = time.time()
    class Mock:
        def __init__(self) -> None:
            pass

        def recreate_edges(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
            edge_start_time = time.time()
            node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
            node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])

            if problem.cost_function == "Geom":
                edges = geom_edges(node_coords)
            elif problem.cost_function == "Euclidean2D":
                edges = euc_2d_edges(node_coords)
            elif problem.cost_function == "Manhatten2D":
                edges = man_2d_edges(node_coords)
            else:
                return "Only Geom, Euclidean2D, and Manhatten2D supported for now."

            logger.info(f"Edge recreation took: {time.time() - edge_start_time:.2f} seconds")
            return edges, node_coords


    # Initialize mock and load data
    init_start_time = time.time()
    mock = Mock()
    load_default_dataset(mock)
    logger.info(f"Initialization and data loading took: {time.time() - init_start_time:.2f} seconds")

    # Problem setup
    selected_ids = [449326, 11294120, 3377349, 22450917, 14512660, 8226374, 9499802, 7076008, 7513156, 4660902, 20430962, 12877927, 26000230, 5984754, 25908049, 13692904, 26298590, 22373916, 15543279, 5280380, 13645260, 13556601, 25715614, 9309546, 24317977, 18464627, 22280699, 8282023, 25057014, 9938953, 477868, 13835989, 11836392, 10242901, 8420142, 9775923, 14735871, 24490423, 8135236, 3051469, 3026975, 22621923, 26766990, 15857202, 18582919, 1811165, 6933768, 6775741, 9429489, 17138658, 24524885, 16682256, 2257519, 4245188, 712487, 16436462, 211165, 15739564, 8406065, 25761819, 13290874, 19493043, 5793089, 16112155, 26447595, 23601187, 22238449, 23566160, 52136, 795746, 14052291, 10157925, 17967748, 13067419, 13567169, 22319948, 7189181, 14845678, 19272752, 23143810, 15153457, 7490529, 23021808, 12599277, 20444769, 7233838, 22550234, 16320250, 9704868, 19555386, 11459214, 12788652, 5940477, 16401094, 13062638, 7372980, 20351425, 13966305, 22451730, 1076547, 17061495, 17727032, 5979348, 26080225, 7631551, 8878966, 16611618, 26222097, 16933983, 10023729, 9169155, 16318728, 22615662, 11301422, 13943722, 8582763, 17265976, 3691662, 22547749, 25369984, 11096388, 24148972, 1105842, 15664416, 9799633, 24706186, 11442282, 4234917, 15282233, 5677393, 2086161, 4585650, 26042565, 13885197, 467533, 18454056, 2652392, 25475182, 3584814, 12530568, 20974661, 11609075, 4650040, 24068785, 19832131, 24859580, 23887856, 19351681, 19568363, 25465150, 12114080, 6416603, 1647554, 3739146, 1144815, 19864847, 14615091, 24293509, 18146600, 20490130, 6268376, 10579211, 15995579, 7805240, 13607888, 7716340, 1563504, 16622953, 1522276, 12237377, 1616173, 20980712, 25763210, 15566835, 21430369, 2891593, 2805740, 8490653, 20983515, 13268244, 18823489, 26619243, 22588513, 24282615, 8450941, 23311385, 16261665, 1301937, 11056834, 26179441, 22683087, 616870, 8600226, 16636381, 13856063, 22204336, 9405917, 13557070, 3206702, 13727561, 17778546, 1130510, 22789958, 2753753, 24498190, 1651762, 10910311, 1272012, 21289395, 25504548, 12290388, 14939723, 23818232, 3947013, 24293454, 5914934, 11410756, 19369169, 6580266, 10088382, 6933103, 23100856, 26609772, 21546609, 19728731, 4564831, 9996037, 16669016, 23950365, 24784575, 25420192, 14003631, 15845751, 5725345, 9761204, 5932330, 21724011, 22421074, 6419137, 188297, 16291898, 4546839, 1202855, 5990547, 25357723, 8115226, 25013068, 8516744, 15615273, 8093324, 13871024, 18813919, 7962584, 5841943, 19641401, 15748953, 18962347, 9518412, 10074369, 8621281, 13536960, 1917285, 4876742, 24861603, 13455066, 15240078, 3673785, 20967251, 20764935, 18542493, 2451359, 18657943, 19626922, 21707666, 5390364, 117853, 2538622, 26172352, 4244235, 13743920, 3245637, 23540849, 24054580, 139885, 14176283, 16822254, 12034558, 13987652, 24022491, 337299, 12970959, 2401623, 5059470, 6593668, 4702292, 17635783, 22307269, 24018398, 1584008, 7041593, 17792813, 15124206, 8744835, 21385297, 5009729, 20116088, 9952112, 19122472, 1508403, 16903980, 17283816, 11219740, 20076255, 1921389, 12877628, 10420282, 9757398, 6107750, 3501081, 24161664, 11624369, 9520466, 18227112, 6409989, 1128391, 5566683, 1060626, 313269, 10127512, 86353, 198911, 17561136, 17745725, 2301751, 4897834, 12406852, 3918584, 4896404, 13324956, 23624026, 23589911, 19989157, 3022032, 25037878, 3336629, 4828014, 885186, 16517946, 26351024, 5208228, 26015548, 3467833, 22277895, 24228195, 14460798, 23588209, 9192255, 26049554, 1085428, 19911032, 10369976, 7581409, 23903527, 17867108, 18534084, 25527149, 18611533, 22756835, 19756592, 15514073, 20777246, 7065076, 2696613, 7225203, 19535033, 11971034, 18621179, 14073276, 3513792, 11204736, 459689, 17620987, 22340195, 511191, 5526624, 6395361, 6083849, 6971358, 12320600, 23787162, 3371197, 11624672, 25758544, 20008342, 15771479, 20785137, 14826404, 16122089, 10398823, 16315868, 22700455, 13880988, 14716146, 104604, 13480734, 13215958, 11616383, 12130468, 5646154, 16504723, 19715872, 15924906, 15290622, 2038587, 8221236, 4021617, 14862702, 11676739, 14105681, 4269753, 21927841, 18340836, 21598882, 8883844, 12185493, 2898768, 18319124, 11298469, 6228508, 16598195, 6679970, 21183103, 18270792, 16139399, 8367501, 1598648, 9925064, 6846891, 14141336, 12832135, 8394160, 24543753, 13941843, 14611633, 24596434, 13504098, 9133271, 10992199, 22731574, 21337947, 1294051, 3545635, 5976763, 393232, 17807014, 20037160, 10650666, 21610428, 18361987, 19316896, 14272583, 15703800, 23659809, 12985773, 24012616, 24775224, 17285776, 11386830, 23367868, 6684569, 18583335, 16598, 16656470, 13296527, 2895431, 7944057, 15585563, 24162255, 713709, 3112724, 12999349, 23075137, 20790091, 13642552, 23194862, 9220580, 21045746, 24978258, 1349195, 11959125, 20639526, 7236884, 4377957, 8182775, 18562812, 19561304, 17863747, 9516063, 17322165, 389786, 4354568, 18048678, 6031682, 17877470, 9258142, 16528308, 23117296, 25069181, 16452796, 25118427, 22122789, 104988, 8402171, 1136253, 20633086, 10570488, 12466755, 3170563, 20834366, 24700585, 174303, 18455107, 230911, 12543890, 3986899, 1828316, 18998656, 8480684, 11420066, 20495912, 24030356, 9921253, 21836273, 24473253, 3136387, 9952292, 10542329, 22654155, 15690987, 10560818, 23267596, 24594395, 20397014, 3508729, 13978396, 17666609, 4378894, 4743671, 705844, 83611, 13368621, 2375912, 20462435, 22706212, 3948438, 21207432, 14390042, 19454356, 2189895, 15429591, 7547441, 19466872, 18927774, 8657889, 18411192, 792821, 16936280, 24092225, 13799788, 4562186, 841297, 7426732, 10170895, 14188048, 12252736, 8948284, 21958979, 17374575, 9816750, 14774689, 23480726, 24133083, 10024118, 9347318, 25104494, 329403, 24879881, 4558882, 17296067, 6461152, 12500224, 6457463, 11211069, 18325946, 26535450, 15757897, 18300280, 2383848, 25372411, 5756702, 22220409, 14370878, 19884379, 19361292, 13024321, 2168021, 6657734, 14927612, 11265731, 15808156, 1726310, 3437774, 9003317, 6381952, 21265626, 20705894, 2440723, 6844903, 24200995, 25931432, 4777058, 6994817, 21045631, 23514864, 21293597, 15030309, 6067022, 1700483, 21389294, 10821959, 21774468, 6843224, 21436364, 14333529, 26142981, 17275627, 26491056, 22381095, 3055119, 13240003, 7344540, 21570133, 13701929, 19835253, 5515151, 11303753, 5056678, 26558832, 6343634, 24680910, 16491663, 9906816, 17051895, 5013696, 2261418, 10223819, 11279758, 6849080, 17329823, 8022779, 18910917, 4279486, 13753349, 16973048, 9223369, 16168489, 9055643, 6981104, 23380191, 15093328, 8618723, 9982878, 5945179, 19704387, 23887653, 2109023, 9333177, 23930783, 666597, 10172730, 24259638, 20874510, 9692637, 11180928, 20780596, 16553523, 5546900, 19574443, 9860415, 3242994, 2628407, 1768647, 20303416, 9336253, 5738859, 15267524, 13813656, 974001, 10182902, 18134983, 14790164, 23693374, 18544598, 13443242, 11635561, 11222751, 18561909, 18897076, 21726658, 4354841, 6368126, 20005285, 1513021, 470513, 19759133, 20409646, 11449865, 26611117, 12864242, 23452144, 10050115, 25174007, 2282803, 23749860, 1617074, 12505652, 17404392, 10372487, 21604520, 9078979, 7895202, 6305797, 3964911, 21253986, 26239814, 25024913, 7627673, 26056547, 25005529, 7230992, 3453090, 22113286, 25475992, 8963247, 6907977, 10526027, 4697811, 8920167, 14298444, 6506315, 17973773, 21935729, 22552061, 14340026, 5737468, 10906253, 6966102, 8962115, 17829976, 15418448, 8761464, 1365077, 1028421, 19022554, 149412, 25886891, 16329294, 1470462, 18267723, 3751686, 21795520, 22196342, 4504787, 14731760, 17293453, 1660757, 11919822, 11333693, 9736602, 11417655, 23316476, 3686408, 19717739, 23835745, 18369738, 12098666, 20600148, 11911064, 13201138, 21176773, 11126393, 3682556, 7536853, 13333806, 6610238, 7815734, 17449211, 292216, 2449508, 7986122, 20362944, 24134261, 435291, 22791595, 17402932, 20140691, 23714080, 3826464, 15336592, 17083201, 260025, 1086351, 4342472, 15405022, 22795661, 11522664, 6280461, 26470245, 19915919, 22508237, 1863419, 9152835, 1794460, 10439808, 14810714, 17554051, 24571822, 2338946, 7924071, 23447959, 7622387, 2994907, 8901807, 10637957, 21869003, 15783412, 20117654, 21958858, 1676311, 21755498, 19300635, 8759621, 13376568, 15140138, 14488864, 19882444, 6612995, 23524266, 3207470, 18676300, 3063238, 2823125, 25644487, 24193657, 26651274, 1760517, 19164119, 6305462, 128437, 3088796, 3270071, 4069512, 10237324, 1657250, 24060711, 5349824, 8636763, 10981489, 15478701, 18844984, 18169703, 2151110, 25426405, 3439933, 6318634, 3927207, 25406902, 17165247, 15408036, 38111, 6563247, 26163908, 5712043, 1498396, 15034473, 16952454, 25997454, 13551620, 6612318, 7432259, 2505493, 21845218, 989090, 4365180, 13842087, 7253211, 2544617, 12109855, 14834498, 26562014, 23051731, 20243883, 19169918, 19469634, 16817579, 12080721, 23329036, 11978396, 18398479, 3563728, 17800688, 2594854, 2185473, 6945453, 22765731, 21899142, 2884803, 4421102, 7945805, 325301, 4971482, 18244683, 15790606, 21890351, 8330883, 743288, 25605314, 8297247, 1674979, 25172824, 24398497, 17116458, 132234, 833766, 11123402, 6144352, 8987996, 24038892, 10445358, 2184413, 13718614, 4436504, 19824265, 8380335, 15702949, 2144633, 20742707, 11025230, 11193448, 16785472, 22139890, 26258462, 4515696, 11312511, 1793854, 11346834, 4173313, 3038660, 7894226, 530888, 16194530, 25377974, 22833473, 7792599, 16643217, 7032938, 2590050, 3091283, 3330752, 20314352, 9750541, 535263, 21177084, 7689089, 8662687, 25966282, 26008544, 12203691, 7887427, 21154253, 16574067, 6884085, 2121130, 24469688, 15644093, 407975, 17720619, 25061165, 7128937, 20710552, 2506959, 16146370, 23930528, 16102146, 6650587, 18750676, 15373484, 12928708, 10159559, 25482169, 4456092, 8418740, 6610145, 9765718, 13401208, 8054397, 16981753, 8506746, 13092581, 6626181, 14336078, 22972444, 3844634, 868149, 15545822, 16078195, 13951072, 2039124, 13642664, 17609693, 18381795, 1964929, 22143982, 11772179, 24947314, 13623549, 24456279, 9313172, 10917107, 23863336, 13101372, 13834977, 8399778, 9307597, 5868533, 13084150, 4787317, 24176410, 26492602, 1170362, 7096876, 24890382, 5981460, 1963296, 25606402, 8504951, 17579846, 26173425, 2686566, 12081693, 26549494, 5432621, 19278493, 25888731, 11958581, 20162616, 2118422]

    setup_start_time = time.time()
    n_nodes = 1057
    m = 10
    test_problem = GraphV2ProblemMulti(
        n_nodes=n_nodes,
        selected_ids=selected_ids,
        dataset_ref="Asia_MSB",
        n_salesmen=m,
        depots=[0] * m,
        cost_function="Geom"
    )
    test_problem.edges, node_coords = mock.recreate_edges(test_problem)
    logger.info(f"Problem setup took:   {time.time() - setup_start_time:.2f} seconds")

    # HGA Solver
    hga_start_time = time.time()
    hga_solver = HGASolver(problem_types=[test_problem])
    route = asyncio.run(hga_solver.solve(test_problem))
    visualize_routes(node_coords, route)
    test_synapse = GraphV2Synapse(problem=test_problem, solution=route)
    score1 = get_multi_minmax_tour_distance(test_synapse)
    logger.info(f"HGA Solver took: {time.time() - hga_start_time:.2f} seconds")

    # Nearest Neighbour Solver
    nn_start_time = time.time()
    solver2 = NearestNeighbourMultiSolver2(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse2 = GraphV2Synapse(problem=test_problem, solution=route2)
    score2 = get_multi_minmax_tour_distance(test_synapse2)
    logger.info(f"Nearest Neighbour Solver took: {time.time() - nn_start_time:.2f} seconds")

    # Final results
    logger.info(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
    print(f"hga scored: {score1} while Multi2 scored: {score2}")