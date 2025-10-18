arg_parser.add_argument("--model-path", default=None, help="The path of the model to benchmark")

assert args.trajs_folder is not None

to_benchmark = ModelPath(
            Path(args.model_path),
            args.prior_only,
            args.prior_nn,
            args.num_steps,
            args.num_save_steps,
            args.trajs_per_protein)

case ModelPath(model_path,
                           prior_only,
                           prior_nn,
                           num_steps,
                           num_save_steps,
                           trajs_per_protein):
                # if model path is a checkpoint, store the model path separately
                if model_path.suffix == ".pth":
                    checkpoint_path = model_path
                    real_model_path = model_path.parent
                else:
                    checkpoint_path = None
                    real_model_path = model_path
                self.benchmark_descriptor = BenchmarkModelPath(
                        checkpoint_path,
                        real_model_path,
                        prior_only,
                        prior_nn,
                        num_steps,
                        num_save_steps,
                        trajs_per_protein)
                
 case BenchmarkModelPath(_, model_path, _, _):
                    output_postfix = model_path.parts[-1]
                    
 match self.benchmark_descriptor:
        model_trajs: list[ModelTraj] = load_model_traj_pickle(model_output_path)
        gen_pickle_path = model_output_path
    
output_path = benchmark_protein()
gen_benchmark.py --traj_path = output_path

def benchmark_protein(self, protein_name: str) -> dict:        
        model_output_path: Path | None = None
        match self.benchmark_descriptor:
            case BenchmarkModelPath(_, model_folder, _, _, num_steps, num_save_steps, trajs_per_protein):
                prior_params = json.load(open(os.path.join(model_folder, "prior_params.json"), "r"))
                random_starting_poses: list[NativeTrajPath] = random.choices(self.starting_poses[protein_name], k=trajs_per_protein)
                logging.debug(f"random_starting_poses: {random_starting_poses} {random_starting_poses[0].pdb_top_path}")
            
                if not self.only_gen_cache:
                    assert generate_trajs_semaphore is not None
                    assert find_gpu_mutex is not None
                    assert gpu_list is not None
                    assert available_gpus is not None
                    with generate_trajs_semaphore:
                        with find_gpu_mutex:
                            gpu_idx = None
                            for i in range(len(gpu_list)):
                                if available_gpus[i]:
                                    available_gpus[i] = False
                                    gpu_idx = i
                                    break
                            assert gpu_idx is not None

                    logging.info(f"starting to run model on protein {protein_name}")
                    # run 10 replicas of the model with simulate.py - also CGed. function returns strings pointing to h5 files 
                    model_output_path = self.run_model(random_starting_poses,
                                                       gpu_list[gpu_idx],
                                                       protein_name,
                                                       num_steps,
                                                       num_save_steps)
                    available_gpus[gpu_idx] = True
                    logging.info(f"finished running model {protein_name}")
                    
                    return model_output_path
                    
def run_model(self,
                  starting_points: list[NativeTrajPath],
                  gpu: int,
                  protein_name: str,
                  num_steps: int,
                  save_steps: int
                  ) -> Path:
        """
        run a simulation on the gpu id at the starting point

        returns a path to the pickle file outputted by by simulate.py
        """
        assert isinstance(self.benchmark_descriptor, BenchmarkModelPath)
        
        traj_path = self.output_dir.joinpath(f"{protein_name}_model_replicas.pkl")
        
        new_envs = os.environ.copy()
        new_envs["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

        logging.debug('===== Will run simulations for protein %s on GPU %d ======' % (protein_name, gpu))

        # need to pass gpu number, the env variable cannot be used anymore
        # prepSim(model_path, [x.pdb_top_path for x in starting_points], temperature=temperature, output=traj_path, steps=1000, save_steps=10, verbose=False, gpu=gpu)
        
        if self.benchmark_descriptor.checkpoint_path is not None:
            modelToRun = self.benchmark_descriptor.checkpoint_path
        else:
            modelToRun = self.benchmark_descriptor.model_folder

        with open(os.path.join(self.log_dir, f"{protein_name}.log"), "w") as outfile:
            cmdList: list[str] = [
                    "./simulate.py",
                    str(modelToRun)
                ] + [x.pdb_top_path for x in starting_points] + [
                    "--temperature", f"{self.temperature}",
                    # 100,000 steps with 20 random starting points should be enough for a good coverage of the TICA space, I tested it. it is very slightly biased towards the starting points, but not by much.
                    "--steps", "%d" % num_steps,  # 100,000 steps should take around 12-15 min on 4 GPUs
                    "--save-steps", "%d" % save_steps,
                    "--output", str(traj_path),
                ]

            if self.benchmark_descriptor.prior_only:
                cmdList.append("--prior-only")

            if self.benchmark_descriptor.prior_nn is not None:
                cmdList.extend(["--prior-nn", str(self.benchmark_descriptor.prior_nn)])

            logging.debug(f"running command \"{' '.join(cmdList)}\"")
            subprocess.run(
                cmdList,
                check=True,
                env=new_envs,
                stdout=outfile,
                stderr=outfile)

        return traj_path