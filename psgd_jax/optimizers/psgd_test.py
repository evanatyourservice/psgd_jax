import numpy as np

from psgd_jax.optimizers.create_optimizer import create_optimizer
from psgd_jax.optimization_experiments import plot_rosenbrock


def main():
    print("Testing PSGD variants on Rosenbrock function")
    all_losses = []
    for use_hessian in [True, False]:
        for precond_type in ["xmat", "uvd", "affine"]:
            steps = 500
            psgd_update_probability = 1.0
            opt, _ = create_optimizer(
                optimizer="psgd",
                learning_rate=0.2 if use_hessian else 0.1,
                min_learning_rate=0.0,
                norm_grads=None,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                nesterov=True,
                weight_decay=0.0,
                lr_schedule="linear",
                schedule_free=False,
                warmup_steps=0,
                total_train_steps=steps,
                gradient_clip=None,
                pmap_axis_name=None,
                graft=False,
                shampoo_precond_every_n=1,
                shampoo_precond_block_size=128,
                psgd_precond_type=precond_type,
                psgd_update_prob=psgd_update_probability,
                psgd_rank=10,
                psgd_heavyball=False,
                psgd_feed_into_adam=True,
                psgd_precond_lr=0.3,
                psgd_precond_init_scale=None,
            )

            hess_name = "Hvp" if use_hessian else "gg^T"
            title = f"{precond_type} PSGD {hess_name}"

            final_loss = plot_rosenbrock(
                optimizer=opt,
                steps=steps,
                plot_title=title,
                psgd_use_hessian=use_hessian,
                psgd_update_probability=psgd_update_probability,
            )
            all_losses.append(final_loss)

    avg_loss = np.mean(all_losses)
    print(f"Average final loss: {avg_loss}")
    return avg_loss


if __name__ == "__main__":
    main()
