#pragma once
namespace tomoto
{
	namespace math
	{


		const double PISQ = __PI * __PI;
		const double HALFPI = 0.5 * __PI;

		const double trunc_schedule[] = { // seq(1,4,by=0.01) -> 301 entries.
		0.64, 0.68, 0.72, 0.75, 0.78, 0.8, 0.83, 0.85, 0.87, 0.89,
		0.91, 0.93, 0.95, 0.96, 0.98,   1, 1.01, 1.03, 1.04, 1.06,
		1.07, 1.09,  1.1, 1.12, 1.13, 1.15, 1.16, 1.17, 1.19,  1.2,
		1.21, 1.23, 1.24, 1.25, 1.26, 1.28, 1.29,  1.3, 1.32, 1.33,
		1.34, 1.35, 1.36, 1.38, 1.39,  1.4, 1.41, 1.42, 1.44, 1.45,
		1.46, 1.47, 1.48,  1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.57,
		1.58, 1.59,  1.6, 1.61, 1.62, 1.63, 1.65, 1.66, 1.67, 1.68,
		1.69,  1.7, 1.71, 1.72, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79,
		1.8 , 1.81, 1.82, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9,
		1.91, 1.92, 1.93, 1.95, 1.96, 1.97, 1.98, 1.99,    2, 2.01,
		2.02, 2.03, 2.04, 2.05, 2.07, 2.08, 2.09,  2.1, 2.11, 2.12,
		2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 2.21, 2.22, 2.23,
		2.24, 2.25, 2.26, 2.27, 2.28, 2.29,  2.3, 2.31, 2.32, 2.33,
		2.35, 2.36, 2.37, 2.38, 2.39,  2.4, 2.41, 2.42, 2.43, 2.44,
		2.45, 2.46, 2.47, 2.48, 2.49, 2.51, 2.52, 2.53, 2.54, 2.55,
		2.56, 2.57, 2.58, 2.59,  2.6, 2.61, 2.62, 2.63, 2.64, 2.65,
		2.66, 2.68, 2.69,  2.7, 2.71, 2.72, 2.73, 2.74, 2.75, 2.76,
		2.77, 2.78, 2.79,  2.8, 2.81, 2.82, 2.83, 2.84, 2.85, 2.87,
		2.88, 2.89,  2.9, 2.91, 2.92, 2.93, 2.94, 2.95, 2.96, 2.97,
		2.98, 2.99,    3, 3.01, 3.02, 3.03, 3.04, 3.06, 3.07, 3.08,
		3.09,  3.1, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18,
		3.19,  3.2, 3.21, 3.22, 3.23, 3.24, 3.25, 3.27, 3.28, 3.29,
		 3.3, 3.31, 3.32, 3.33, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39,
		 3.4, 3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.49,  3.5,
		3.51, 3.52, 3.53, 3.54, 3.55, 3.56, 3.57, 3.58, 3.59,  3.6,
		3.61, 3.62, 3.63, 3.64, 3.65, 3.66, 3.67, 3.68, 3.69, 3.71,
		3.72, 3.73, 3.74, 3.75, 3.76, 3.77, 3.78, 3.79,  3.8, 3.81,
		3.82, 3.83, 3.84, 3.85, 3.86, 3.87, 3.88, 3.89,  3.9, 3.91,
		3.92, 3.93, 3.95, 3.96, 3.97, 3.98, 3.99,    4, 4.01, 4.02,
		4.03, 4.04, 4.05, 4.06, 4.07, 4.08, 4.09,  4.1, 4.11, 4.12, 4.13 };

		//------------------------------------------------------------------------------

		class PolyaGammaAlt
		{

		public:

			// Draw.
			double draw(double h, double z, RNG& r, int max_inner = 200)
			{
				assert(h >= 1);

				double n = floor((h - 1.0) / 4.0);
				double remain = h - 4.0 * n;

				double x = 0.0;

				for (int i = 0; i < (int)n; i++)
					x += draw_abridged(4.0, z, r);
				if (remain > 4.0)
					x += draw_abridged(0.5 * remain, z, r) + draw_abridged(0.5 * remain, z, r);
				else
					x += draw_abridged(remain, z, r);

				return x;
			}

			double draw_abridged(double h, double z, RNG& r, int max_inner = 200)
			{
				assert(h >= 1 && h <= 4);
				// Change the parameter.
				z = fabs(z) * 0.5;

				int    idx = (int)floor((h - 1.0)*100.0);
				double trunc = trunc_schedule[idx];

				// Now sample 0.25 * J^*(1, z := z/2).
				double rate_z = 0.125 * __PI*__PI + 0.5 * z*z;
				double weight_left = w_left(trunc, h, z);
				double weight_right = w_right(trunc, h, z);
				double prob_right = weight_right / (weight_right + weight_left);

				// printf("prob_right: %g\n", prob_right);

				double coef1_h = exp(h * log(2.0) - 0.5 * log(2.0 * __PI));
				// double gamma_nh_over_n = RNG::Gamma(h);
				double gnh_over_gn1_gh = 1.0; // Will fill in value on first call to a_coef_recursive.

				int num_trials = 0;
				int total_iter = 0;

				while (num_trials < 10000) {
					num_trials++;

					double X = 0.0;
					double Y = 0.0;

					// if (r.unif() < p/(p+q))
					double uu = std::generate_canonical<double, 64>(r);
					if (uu < prob_right)
						X = r.ltgamma(h, rate_z, trunc);
					else
						X = rtigauss(h, z, trunc, r);

					// double S  = a_coef(0, X, h);
					double S = a_coef_recursive(0.0, X, h, coef1_h, gnh_over_gn1_gh);
					double a_n = S;
					// double a_n2 = S2;
					// printf("a_n=%g, a_n2=%g\n", a_n, a_n2);
					double gt = g_tilde(X, h, trunc);
					Y = std::generate_canonical<double, 64>(r) * gt;

					// printf("test gt: %g\n", g_tilde(trunc * 0.1, h, trunc));
					// printf("X, Y, S, gt: %g, %g, %g, %g\n", X, Y, S, gt);

					bool decreasing = false;

					int  n = 0;
					bool go = true;

					// Cap the number of iterations?
					while (go && n < max_inner) {
						total_iter++;

						++n;
						double prev = a_n;
						// a_n  = a_coef(n, X, h);
						a_n = a_coef_recursive((double)n, X, h, coef1_h, gnh_over_gn1_gh);
						// printf("a_n=%g, a_n2=%g\n", a_n, a_n2);
						decreasing = a_n <= prev;

						if (n % 2 == 1) {
							S = S - a_n;
							if (Y <= S && decreasing) return 0.25 * X;
						}
						else {
							S = S + a_n;
							if (Y > S && decreasing) go = false;
						}

					}
					// Need Y <= S in event that Y = S, e.g. when X = 0.

				}

				// We should never get here.
				return -1.0;
			} // draw

			// Helper.
			double a_coef(int n, double x, double h)
			{
				double d_n = 2.0 * (double)n + h;
				double log_out = h * log(2.0) - std::lgamma(h) + std::lgamma(n + h)
					- std::lgamma(n + 1) + log(d_n)
					- 0.5 * log(2.0 * __PI * x * x * x) - 0.5 * d_n * d_n / x;
				double out = exp(log_out);
				// double out = exp(out) is a legal command.  Weird.
				return out;
			}

			double a_coef_recursive(double n, double x, double h, double coef_h, double& gamma_nh_over_n)
			{
				double d_n = 2.0 * (double)n + h;
				// gamma_nh_over_n *= (n + h - 1) / n;  // Can speed up further by separate function for a0 and an, n > 0.
				if (n != 0)
					gamma_nh_over_n *= (n + h - 1) / n;
				else
					gamma_nh_over_n = 1.0;
				double coef = coef_h * gamma_nh_over_n;
				double log_kernel = -0.5 * (log(x * x * x) + d_n * d_n / x) + log(d_n);
				return coef * exp(log_kernel);
				// double out = exp(out) is a legal command.  Weird.
			}

			double g_tilde(double x, double h, double trunc)
			{
				double out = 0;
				if (x > trunc)
					out = exp(h * log(0.5 * __PI) + (h - 1) * log(x) - PISQ * 0.125 * x - std::lgamma(h));
				else
					out = h * exp(h * log(2.0) - 0.5 * log(2.0 * __PI * x * x * x) - 0.5 * h * h / x);
				// out = h * pow(2, h) * pow(2 * __PI * pow(x,3), -0.5) * exp(-0.5 * pow(h,2) / x);
				return out;
			}

			double pigauss(double x, double z, double lambda)
			{
				// z = 1 / mean
				double b = sqrt(lambda / x) * (x * z - 1);
				double a = sqrt(lambda / x) * (x * z + 1) * -1.0;
				double y = p_norm(b) + exp(2 * lambda * z) * p_norm(a);
				return y;
			}

			double rtigauss(double h, double z, double trunc, RNG& r)
			{
				z = fabs(z);
				double mu = h / z;
				double X = trunc + 1.0;
				if (mu > trunc) { // mu > t
					double alpha = 0.0;
					while (std::generate_canonical<double, 64>(r) > alpha) {
						X = rtinvchi2(h, trunc, r);
						alpha = exp(-0.5 * z*z * X);
					}
					// printf("rtigauss, part i: %g\n", X);
				}
				else {
					while (X > trunc) {
						X = r.igauss(mu, h*h);
					}
					// printf("rtigauss, part ii: %g\n", X);
				}
				return X;
			}

			double w_left(double trunc, double h, double z)
			{
				double out = 0;
				if (z != 0)
					out = exp(h * (log(2.0) - z)) * pigauss(trunc, z / h, h*h);
				else
					out = exp(h * log(2.0)) * (1.0 - RNG::p_gamma_rate(1 / trunc, 0.5, 0.5*h*h));
				return out;
			}

			double w_right(double trunc, double h, double z)
			{
				double lambda_z = PISQ * 0.125 + 0.5 * z * z;
				double p = exp(h * log(HALFPI / lambda_z)) * (1.0 - RNG::p_gamma_rate(trunc, h, lambda_z));
				return p;
			}

		};

		double rtinvchi2(double h, double trunc, RNG& r)
		{
			double h2 = h * h;
			double R = trunc / h2;
			double X = 0.0;
			// I need to consider using a different truncated normal sampler.
			double E1 = std::exponential_distribution<>()(r); double E2 = std::exponential_distribution<>()(r);
			while ((E1*E1) > (2 * E2 / R)) {
				// printf("E %g %g %g %g\n", E1, E2, E1*E1, 2*E2/R);
				E1 = std::exponential_distribution<>()(r); E2 = std::exponential_distribution<>()(r);
			}
			// printf("E %g %g \n", E1, E2);
			X = 1 + E1 * R;
			X = R / (X * X);
			X = h2 * X;
			return X;
		}



		struct FD {
			double val;
			double der;
		};

		struct Line {
			double slope;
			double icept;
		};

		// PolyaGamma approximation by SP.
		class PolyaGammaSP
		{

		public:

			int draw(double& d, double h, double z, RNG& r, int maxiter = 200)
			{
				if (n < 1) fprintf(stderr, "PolyaGammaSP::draw: n must be >= 1.\n");
				z = 0.5 * fabs(z);

				double xl = y_func(-1 * z*z);    // Mode of phi - Left point.
				double md = xl * 1.1;          // Mid point.
				double xr = xl * 1.2;          // Right point.

				// printf("xl, md, xr: %g, %g, %g\n", xl, md, xr);

				// Inflation constants
				// double vmd  = yv.v_func(md);
				double vmd = v_eval(md);
				double K2md = 0.0;

				if (fabs(vmd) >= 1e-6)
					K2md = md * md + (1 - md) / vmd;
				else
					K2md = md * md - 1 / 3 - (2 / 15) * vmd;

				double m2 = md * md;
				double al = m2 * md / K2md;
				double ar = m2 / K2md;

				// printf("vmd, K2md, al, ar: %g, %g %g %g\n", vmd, K2md, al, ar);

				// Tangent lines info.
				Line ll, lr;
				tangent_to_eta(xl, z, md, ll);
				tangent_to_eta(xr, z, md, lr);

				double rl = -1. * ll.slope;
				double rr = -1. * lr.slope;
				double il = ll.icept;
				double ir = lr.icept;

				// printf("rl, rr, il, ir: %g, %g, %g, %g\n", rl, rr, il, ir);

				// Constants
				double lcn = 0.5 * log(0.5 * n / __PI);
				double rt2rl = sqrt(2 * rl);

				// printf("sqrt(rl): %g\n", rt2rl);

				// Weights
				double wl, wr, wt, pl;

				wl = exp(0.5 * log(al) - n * rt2rl + n * il + 0.5 * n * 1. / md) *
					RNG::p_igauss(md, 1. / rt2rl, n);

				wr = exp(0.5 * log(ar) + lcn - n * log(n * rr) + n * ir - n * log(md)) *
					// yv.upperIncompleteGamma(md, n, n*rr);
					std::lgamma(n) * (1.0 - RNG::p_gamma_rate(md, n, n*rr));

				// printf("wl, wr: %g, %g\n", wl, wr);

				wt = wl + wr;
				pl = wl / wt;

				// Sample
				bool go = true;
				int iter = 0;
				double X = 2.0;
				double F = 0.0;

				while (go && iter < maxiter) {
					// Put first so check on first pass.
#ifdef USE_R
					if (iter % 1000 == 0) R_CheckUserInterrupt();
#endif

					iter++;

					double phi_ev;
					if (std::generate_canonical<double, 64>(r) < pl) {
						X = rtigauss(1. / rt2rl, n, md, r);
						phi_ev = n * (il - rl * X) + 0.5 * n * ((1. - 1. / X) - (1. - 1. / md));
						F = exp(0.5 * log(al) + lcn - 1.5 * log(X) + phi_ev);
					}
					else {
						X = r.ltgamma(n, n*rr, md);
						phi_ev = n * (ir - rr * X) + n * (log(X) - log(md));
						F = exp(0.5 * log(ar) + lcn + phi_ev) / X;
					}

					double spa = sp_approx(X, n, z);

					if (F * std::generate_canonical<double, 64>(r) < spa) go = false;

				}

				// return n * 0.25 * X;
				d = n * 0.25 * X;
				return iter;
			}

		protected:

			// Helper.

			double w_left(double trunc, double h, double z);
			double w_right(double trunc, double h, double z);

			void   delta_func(double x, double mid, FD& delta)
			{
				if (x >= mid) {
					delta.val = log(x) - log(mid);
					delta.der = 1.0 / x;
				}
				else {
					delta.val = 0.5 * (1 - 1.0 / x) - 0.5 * (1 - 1.0 / mid);
					delta.der = 0.5 / (x*x);
				}
			}

			double phi_func(double x, double z, FD& phi)
			{
				// double v = yv.v_func(x);
				double v = v_eval(x);
				double u = 0.5 * v;
				double t = u + 0.5 * z*z;

				phi.val = log(cosh(fabs(z))) - log(cos_rt(v)) - t * x;
				phi.der = -1.0 * t;

				return v;
			}

			double tangent_to_eta(double x, double z, double mid, Line& tl)
			{
				FD phi, delta, eta;
				double v;

				v = phi_func(x, z, phi);
				delta_func(x, mid, delta);

				eta.val = phi.val - delta.val;
				eta.der = phi.der - delta.der;

				// printf("v=%g\nphi=%g, phi.d=%g\ndelta=%g, delta.d=%g\neta=%g, eta.d=%g\n",
				// 	 v, phi.val, phi.der, delta.val, delta.der, eta.val, eta.der);

				tl.slope = eta.der;
				tl.icept = eta.val - eta.der * x;

				return v;
			}

			double sp_approx(double x, double n, double z)
			{
				// double v  = yv.v_func(x);
				double v = v_eval(x);
				double u = 0.5 * v;
				double z2 = z * z;
				double t = u + 0.5 * z2;
				// double m  = y_func(-1 * z2);

				double phi = log(cosh(z)) - log(cos_rt(v)) - t * x;

				double K2 = 0.0;
				if (fabs(v) >= 1e-6)
					K2 = x * x + (1 - x) / v;
				else
					K2 = x * x - 1 / 3 - (2 / 15) * v;

				double log_spa = 0.5 * log(0.5 * n / __PI) - 0.5 * log(K2) + n * phi;
				return exp(log_spa);
			}

			double cos_rt(double v)
			{
				double y = 0.0;
				double r = sqrt(fabs(v));
				if (v >= 0)
					y = cos(r);
				else
					y = cosh(r);
				return y;
			}

			// YV yv;

			double rtigauss(double mu, double lambda, double trunc, RNG& r)
			{
				// mu = fabs(mu);
				double X = trunc + 1.0;
				if (trunc < mu) { // mu > t
					double alpha = 0.0;
					while (std::generate_canonical<double, 64>(r) > alpha) {
						X = r.rtinvchi2(lambda, trunc);
						alpha = exp(-0.5 * lambda / (mu*mu) * X);
					}
					// printf("rtigauss, part i: %g\n", X);
				}
				else {
					while (X > trunc) {
						X = r.igauss(mu, lambda);
					}
					// printf("rtigauss, part ii: %g\n", X);
				}
				return X;
			}

			double y_func(double v) // y = tan(sqrt(v)) / sqrt(v);
			{
				double tol = 1e-6;
				double y = 0.0;
				double r = sqrt(fabs(v));
				if (v > tol)
					y = tan(r) / r;
				else if (v < -1 * tol)
					y = tanh(r) / r;
				else
					y = 1 + (1 / 3) * v + (2 / 15) * v * v + (17 / 315) * v * v * v;
				return y;
			}

		};


		class PolyaGammaSmallB
		{
		public:
			PolyaGammaSmallB() {}

			// Draw.
			double draw(double b, double z, RNG& r)
			{
				double x;
				if (z == 0)
				{
					x = draw_invgamma_rej(b, r) / 4.0;
				}
				else
				{
					x = draw_invgauss_rej(b, z / 2.0, r) / 4.0;
				}
				return x;
			}

		private:

			double draw_invgauss_rej(double b, double z, RNG& r)
			{
				bool success = false;
				int niter = 0;

				//    fprintf(stderr, "b: %.3f\t z: %.3f\n", b, z);
				double mu = b / fabs(z);
				double lambda = b * b;

				double x, u;

				while (!success && niter < 100)
				{
					x = r.igauss(mu, lambda);
					u = std::generate_canonical<double, 64>(r);
					if (u < one_minus_psi(x, b))
					{
						success = true;
					}
					niter += 1;
				}

				if (!success)
				{
					throw std::runtime_error("InvGauss rejection sampler failed for MAXITER iterations.");
				}

				return x;
			}

			double draw_invgamma_rej(double b, RNG& r)
			{
				bool success = false;
				int niter = 0;

				double alpha = 0.5;
				double beta = b * b / 2.0;

				double x, u;

				while (!success && niter < 100)
				{
					x = r.igamma(alpha, beta);
					u = std::generate_canonical<double, 64>(r);
					if (u < one_minus_psi(x, b))
					{
						success = true;
					}
					niter += 1;
				}

				if (!success)
				{
					throw std::runtime_error("InvGamma rejection sampler failed for MAXITER iterations.");
				}

				return x;
			}

			// Helper.
			inline double one_minus_psi(double x, double b)
			{
				double omp = 1.0;
				omp -= (2.0 + b) * exp(-2.*(b + 1.0) / x);
				omp += (1.0 + b)*(4.0 + b) / 2.0 * exp(-4.0*(b + 2.0) / x);
				omp -= (2.0 + b)*(1.0 + b)*(6.0 + b) / 6.0 * exp(-6.0*(b + 3.0) / x);
				omp += (3.0 + b)*(2.0 + b)*(1.0 + b)*(8.0 + b) / 24.0 * exp(-8.0*(b + 4.0) / x);
				omp -= (4.0 + b)*(3.0 + b)*(2.0 + b)*(1.0 + b)*(10.0 + b) / 120.0 * exp(-10.0*(b + 5.0) / x);
				return omp;
			}

		};

		template <typename Real>
		class PolyaGammaHybrid
		{
		private:
			RNG*          rng;

		public:
			// Constructor and destructor
			PolyaGammaHybrid(unsigned long seed);
			~PolyaGammaHybrid();


			PolyaGamma       dv;
			PolyaGammaAlt    al;
			PolyaGammaSP     sp;
			PolyaGammaSmallB sb;

			void set_trunc(int trunc);
			Real draw(Real b, Real z);

		};

		// Constructor and Destructor
		template <typename Real>
		PolyaGammaHybrid<Real>::PolyaGammaHybrid(unsigned long seed)
		{
			rng = new RNG(seed);
		}

		template <typename Real>
		PolyaGammaHybrid<Real>::~PolyaGammaHybrid()
		{
			delete rng;
		}

		// Plumbing
		template <typename Real>
		void PolyaGammaHybrid<Real>::set_trunc(int trunc)
		{
			dv.set_trunc(trunc);
		}

		// Draw
		template <typename Real>
		Real PolyaGammaHybrid<Real>::draw(Real b_, Real z_)
		{
			double x;

			double b = (double)b_;
			double z = (double)z_;

			if (b > 170)
			{
				double m = dv.pg_m1(b, z);
				double v = dv.pg_m2(b, z) - m * m;
				x = (Real)rng->norm(m, sqrt(v));
			}
			else if (b > 13)
			{
				sp.draw(x, b, z, *rng);
			}
			else if (b == 1 || b == 2)
			{
				x = dv.draw((int)b, z, *rng);
			}
			else if (b > 1)
			{
				x = al.draw(b, z, *rng);
			}
			else if (b > 0)
			{
				x = sb.draw(b, z, *rng);
			}
			else
			{
				x = 0.0;
			}

			return (Real)x;
		}
	}
}