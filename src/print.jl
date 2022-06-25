function solver_info(solver)
    println(crayon"bold red","
    __  __      _               _
   |  \\/  |    | |             | |
   | \\  / | ___| |__  _ __ ___ | |_ _ __ __ _
   | |\\/| |/ _ \\ '_ \\| '__/ _ \\| __| '__/ _` |
   | |  | |  __/ | | | | | (_) | |_| | | (_| |
   |_|  |_|\\___|_| |_|_|  \\___/ \\__|_|  \\__,_|
    ")

    println(crayon"reset bold black",
    "              Simon Le Cleac'h & Taylor Howell")
    println("                       Robotic Exploration Lab")
    println("                           Stanford University \n")
    print(crayon"reset")
end

function iteration_status(
        iterations,
        equality_violation,
        cone_product_violation,
        central_path,
        step_size)

    # header
    if rem(iterations - 1, 10) == 0
        @printf "-------------------------------------------------------------------\n"
        @printf "iter  |equality|   |comp|      central path    step  \n"
        @printf "-------------------------------------------------------------------\n"
    end

    # iteration information
    @printf("%3d   %9.2e    %9.2e  %9.2e       %9.2e \n",
        iterations,
        equality_violation,
        cone_product_violation,
        central_path,
        step_size)
end

function solver_status(solver, status)
    @printf "------------------------------------------------------------------------------------------------\n"
    println("solution gradients: $(solver.options.differentiate)")
    println("solve status:       $(status ? "success" : "failure")")
    solver.dimensions.variables < 10 && println("solution:           $(round.(solver.solution.all, sigdigits=3))")
    @printf "------------------------------------------------------------------------------------------------\n"
end
