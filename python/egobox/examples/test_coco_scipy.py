import cocoex  # experimentation module
import cocopp  # post-processing module (not strictly necessary)
import scipy  # to define the solver to be benchmarked

### input
suite_name = "bbob"
fmin = scipy.optimize.fmin  # optimizer to be benchmarked
budget_multiplier = 1  # x dimension, increase to 3, 10, 30,...

### prepare
suite = cocoex.Suite(
    suite_name, "", ""
)  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
output_folder = "{}_of_{}_{}D_on_{}".format(
    fmin.__name__, fmin.__module__ or "", int(budget_multiplier), suite_name
)
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
repeater = cocoex.ExperimentRepeater(budget_multiplier)  # 0 == no repetitions
minimal_print = cocoex.utilities.MiniPrint()

### go
while not repeater.done():  # while budget is left and successes are few
    for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
        if repeater.done(problem):
            continue  # skip this problem
        problem.observe_with(observer)  # generate data for cocopp
        problem(problem.dimension * [0])  # for better comparability
        xopt = fmin(problem, repeater.initial_solution_proposal(problem), disp=False)
        problem(xopt)  # make sure the returned solution is evaluated
        repeater.track(problem)  # track evaluations and final_target_hit
        minimal_print(problem)  # show progress

### post-process data
# cocopp.main(observer.result_folder + ' bfgs!');  # re-run folders look like "...-001" etc
