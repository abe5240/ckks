import random
import numpy as np


class Polynomial:
    """Polynomial arithmetic in Z[X]/(X^N+1) and Z_Q[X]/(X^N+1)."""
    
    def __init__(self, coeffs, degree=None, modulus=None):
        self.modulus = modulus
        self.degree = degree or len(coeffs)

        if len(coeffs) > self.degree:
            raise ValueError(
                f"Number of coefficients ({len(coeffs)}) "
                f"exceeds specified degree ({self.degree})"
            )
            
        # Pad coefficients with zeros if needed
        padded_coeffs = np.zeros(self.degree, dtype=object)
        padded_coeffs[:len(coeffs)] = coeffs
        
        # Convert to polynomial object
        self.poly = np.polynomial.Polynomial(padded_coeffs)

    @property
    def coeffs(self):
        return self.poly.coef

    def __getitem__(self, idx):
        return self.poly.coef[idx]

    def __len__(self):
        return self.degree

    def __repr__(self):
        bits = self.modulus.bit_length() if self.modulus else "N/A"
        return f"Polynomial(degree={self.degree}, modulus_bits={bits})"

    def __neg__(self):
        return self.copy(coeffs=-self.coeffs)

    def __add__(self, other):
        if not isinstance(other, (int, Polynomial)):
            return NotImplemented
        
        # Case 1: Polynomial + Integer
        # Integer is treated as a constant polynomial (x^0 term)
        if isinstance(other, int):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += other

            if self.modulus is not None:
                new_coeffs[0] %= self.modulus
                
            return self.copy(coeffs=new_coeffs)
        
        # Case 2: Polynomial + Polynomial
        # Use the minimum of the two moduli or None
        modulus_choices = filter(None, [self.modulus, other.modulus])
        modulus = min(modulus_choices, default=None)
        
        # Add the polynomials (mod Q if it exists)
        res = self.poly + other.poly
        if modulus is not None:
            coeffs = res.coef % modulus
            return self.copy(coeffs=coeffs, modulus=modulus)
        
        return self.copy(coeffs=res.coef)

    def __sub__(self, other):
        return self.__add__(-other)
        
    def __mul__(self, other):
        if not isinstance(other, (int, Polynomial)):
            return NotImplemented
        
        # Case 1: Polynomial * Integer
        if isinstance(other, int):
            if self.modulus is None:
                return self.copy(coeffs=self.coeffs * other)

            new_coeffs = (self.coeffs * other) % self.modulus
            return self.copy(coeffs=new_coeffs)

        # Case 2: Polynomial * Polynomial
        modulus_choices = filter(None, [self.modulus, other.modulus])
        modulus = min(modulus_choices, default=None)

        # Perform standard polynomial multiplication (deg. 2N-2)
        product_coeffs = (self.poly * other.poly).coef
                
        # Reduce modulo X^N + 1 (negacyclic convolution)
        result_coeffs = np.zeros(self.degree, dtype=object) # (deg. N-1)
        for i, coeff in enumerate(product_coeffs):
            pos = i % self.degree
            sign = -1 if i >= self.degree else 1
            result_coeffs[pos] += sign * coeff

        # And reduce mod Q if need be
        if modulus is not None:
            result_coeffs = result_coeffs % modulus
            
        return self.copy(coeffs=result_coeffs, modulus=modulus)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def automorphism(self, k):
        """Apply automorphism φ_k: X -> X^k."""
        new_coeffs = np.zeros(self.degree, dtype=object)
        
        for power, coeff in enumerate(self.poly.coef):
            new_power = (power * k) % (2 * self.degree) # i * 5^r 
            new_pos = new_power % self.degree
            sign = -1 if new_power >= self.degree else 1

            val = new_coeffs[new_pos] + sign * coeff
            if self.modulus:
                val %= self.modulus
            new_coeffs[new_pos] = val

        return self.copy(coeffs=new_coeffs)

    def modswitch(self, target_modulus: int):
        """Change polynomial modulus."""
        coeffs = self.coeffs % target_modulus
        return Polynomial(coeffs=coeffs, degree=self.degree, modulus=target_modulus)

    def moddown(self, scalar: int):
        """Scale down polynomial coefficients by rounding."""
        if self.modulus is None:
            raise ValueError("Modulus must be set to perform moddown.")
        
        # Safe round(coeffs / scalar) for large integers.
        scaled_coeffs = [(c + (scalar // 2)) // scalar for c in self.poly.coef]
        scaled_modulus = self.modulus // scalar
        return self.copy(coeffs=scaled_coeffs, modulus=scaled_modulus)
    
    def copy(self, **kwargs):
        defaults = {
            'coeffs': self.coeffs, 
            'degree': self.degree, 
            'modulus': self.modulus,
        }
        return Polynomial(**{**defaults, **kwargs})


class KeyChain:
    """Manages cryptographic keys for the homomorphic encryption scheme."""
    
    def __init__(self, degree, hamming_weight, modulus):
        self.degree = degree
        self.hamming_weight = hamming_weight
        self.modulus = modulus
        
        self.sk = self.generate_secret_key() 
        self.pk = self.generate_public_key() 
        
        self.rlk = None
        self.swks = {}

    def ensure_relin_key(self):
        if self.rlk is None:
            self.rlk = self._generate_relin_key()

    def ensure_rotation_key(self, galois_element):
        if galois_element not in self.swks:
            self.swks[galois_element] = self._generate_rotation_key(galois_element)

    def generate_secret_key(self):
        """Generate sparse secret key with ±1 coefficients using numpy."""
        coeffs = np.zeros(self.degree, dtype=object)
        indices = np.random.choice(self.degree, size=self.hamming_weight, replace=False)
        coeffs[indices] = np.random.choice([-1, 1], size=self.hamming_weight)
        
        return Polynomial(coeffs=coeffs, modulus=None) 

    def generate_public_key(self):
        """Generate public key from the secret key."""
        a = self.generate_uniform_noise()
        e = self.generate_gaussian_noise()
        
        # Effectively an encryption of zero under the secret key
        pk0 = -a * self.sk + e
        pk1 = a

        return PublicKey(pk0=pk0, pk1=pk1, modulus=self.modulus)
    
    def generate_gaussian_noise(self, stdev=3.2, modulus=None):
        """Generates a noise vector from a discrete Gaussian distribution."""
        modulus = modulus or self.modulus
        noise = [round(np.random.normal(0, stdev)) for _ in range(self.degree)]
        return Polynomial(coeffs=noise, modulus=modulus)
    
    def generate_uniform_noise(self, modulus=None):
        modulus = modulus or self.modulus
        uniform = [random.randrange(modulus) for _ in range(self.degree)]
        return Polynomial(coeffs=uniform, modulus=modulus)
    
    def _generate_switching_key(self, target_poly, base_modulus):
        """Generates a generic key for switching."""
        extended_mod = base_modulus ** 2
        a = self.generate_uniform_noise(modulus=extended_mod)
        e = self.generate_gaussian_noise(modulus=extended_mod)

        sk_qq = self.sk.modswitch(extended_mod)
        swk0 = -a * sk_qq + target_poly + e
        swk1 = a
        
        return SwitchingKey(swk0=swk0, swk1=swk1, modulus=extended_mod)

    def _generate_relin_key(self):
        """Generates the relinearization key. Encrypts Q*s^2."""
        # Relinearization key is generated relative to the top-level modulus
        qs2 = self.modulus * self.sk * self.sk
        return self._generate_switching_key(qs2, self.modulus)
    
    def _generate_rotation_key(self, galois_element):
        """Generates a rotation key. Encrypts Q*φ_k(s)."""
        # Rotation keys are also generated relative to the top-level modulus
        qsk_perm = self.modulus * self.sk.automorphism(galois_element)
        return self._generate_switching_key(qsk_perm, self.modulus)


class Engine:
    """Main encryption engine implementing CKKS-style homomorphic encryption."""
    
    def __init__(self, degree, modulus_bits, precision_bits, hamming_weight=16):
        self.degree = degree
        self.modulus = 1 << modulus_bits 
        self.precision = 1 << precision_bits
        self.hamming_weight = min(hamming_weight, self.degree)
        self.U = self._generate_vandermonde()
        self.keychain = KeyChain(self.degree, self.hamming_weight, self.modulus)

    @property
    def n_slots(self):
        return self.degree // 2

    def _generate_vandermonde(self):
        """Generate Vandermonde matrix for encoding/decoding."""
        w = np.exp(1j * np.pi / self.degree, dtype=np.complex128)
        exponents = [pow(5, j, 2 * self.degree) for j in range(self.n_slots)]
        exponents = np.array(exponents, dtype=np.complex128)
        k = np.arange(self.degree)
        return w ** (exponents[:, None] * k)

    def encode(self, cleartext, simd=True):
        """Encode a vector into a plaintext polynomial."""
        capacity = self.n_slots if simd else self.degree
        
        cleartext_arr = np.array(cleartext, dtype=object)
        if cleartext_arr.size > capacity:
            raise ValueError(f"Input vector exceeds capacity {capacity}")
        
        # Pad the end of the cleartext with zeros if need be
        padded_cleartext = np.zeros(capacity, dtype=object)
        padded_cleartext[:cleartext_arr.size] = cleartext_arr
        
        if not simd:
            coeffs = [round(x * self.precision) for x in padded_cleartext]
            return Plaintext(
                Polynomial(coeffs, modulus=None), 
                self.precision, 
                simd
            )

        slots = [x * self.precision for x in padded_cleartext]
        slots_conj = [x.conjugate() for x in slots]

        coeffs = (self.U.conj().T @ slots + self.U.T @ slots_conj).real / self.degree
        coeffs = [round(coef) for coef in coeffs]

        return Plaintext(
            Polynomial(coeffs=coeffs, modulus=None), 
            self.precision, 
            simd
        )
    
    def decode(self, plaintext):
        """Decode a plaintext polynomial back to a vector."""
        if not plaintext.simd:
            return (plaintext.poly.coeffs / plaintext.scale_factor).astype(np.complex128)
        
        slots = (self.U @ plaintext.poly.coeffs) / plaintext.scale_factor
        return slots.astype(np.complex128)

    def encrypt(self, plaintext):
        """Encrypt a plaintext polynomial using the public key."""
        v  = self.keychain.generate_secret_key() # Ephemeral key
        e0 = self.keychain.generate_gaussian_noise()
        e1 = self.keychain.generate_gaussian_noise()
        pk = self.keychain.pk

        c0 = v * pk.pk0 + plaintext.poly + e0 
        c1 = v * pk.pk1 + e1

        return Ciphertext(
            c0, c1, 
            self.precision, 
            plaintext.scale_factor, 
            self.modulus, 
            self.keychain, 
            plaintext.simd
        )

    def decrypt(self, ciphertext):
        """Decrypt a ciphertext using the secret key (vectorized)."""
        m_poly = ciphertext.c0 + ciphertext.c1 * self.keychain.sk
        
        q_half = ciphertext.modulus >> 1
        coeffs = m_poly.coeffs.copy()
       
        # Bring coefficients from [0, Q-1] to [-Q/2, Q/2)
        coeffs[coeffs > q_half] -= ciphertext.modulus
        
        return Plaintext(
            Polynomial(coeffs=coeffs, degree=self.degree), 
            ciphertext.scale_factor, 
            ciphertext.simd
        )


class Plaintext:
    def __init__(self, poly, scale_factor, simd):
        if poly.modulus is not None:
            raise ValueError(
                "Plaintext polynomial must live in Z[X], not Z_q[X]. " \
                "Modulus must be None."
            )    
        self.poly = poly
        self.scale_factor = scale_factor 
        self.simd = simd

    @property
    def degree(self):
        return self.poly.degree
    
    def __neg__(self):
        return Plaintext(-self.poly, self.scale_factor, self.simd)
    
    def __repr__(self):
        return f"Plaintext(degree={len(self.poly)}, scale={self.scale_factor}"


class Ciphertext:
    def __init__(self, c0, c1, precision, scale_factor, modulus, keychain, simd):
        self.c0 = c0
        self.c1 = c1
        self.precision = precision
        self.scale_factor = scale_factor
        self.modulus = modulus
        self.keychain = keychain
        self.simd = simd

    @property
    def degree(self):
        return self.c0.degree
    
    @property
    def level(self):
        return round(np.log2(self.keychain.modulus / self.modulus))

    def __repr__(self):
        scale_bits = round(np.log2(self.scale_factor))
        mod_bits = self.modulus.bit_length()
        return f"Ciphertext(level={self.level}, mod_bits={mod_bits}, " \
               f"scale_bits={scale_bits})"
        
    def __neg__(self):
        return self.copy(c0=-self.c0, c1=-self.c1)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            # Add an integer to this ciphertext
            c0_new = self.c0 + round(other * self.scale_factor)
            return self.copy(c0=c0_new, c1=self.c1)
        
        assert self.scale_factor == other.scale_factor

        if isinstance(other, Plaintext):
            # Add a plaintext to this ciphertext
            c0_new = self.c0 + other.poly
            return self.copy(c0=c0_new, c1=self.c1)
        
        if isinstance(other, Ciphertext):
            # Add two ciphertexts
            c0_new = self.c0 + other.c0
            c1_new = self.c1 + other.c1
            return self.copy(c0=c0_new, c1=c1_new)
       
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __mul__(self, other):
        # Case 1: Ciphertext-Integer/Float/Plaintext multiplication
        if isinstance(other, (int, float, Plaintext)):
            # These don't change the level, so we handle them simply.
            if isinstance(other, (int, float)):
                pt = Plaintext(
                    Polynomial(coeffs=[round(other * self.precision)]),
                    self.precision,
                    self.simd
                )
            else: # is Plaintext
                pt = other

            c0_new = self.c0 * pt.poly 
            c1_new = self.c1 * pt.poly 
            scale_new = self.scale_factor * pt.scale_factor 
            
            res = self.copy(c0=c0_new, c1=c1_new, scale_factor=scale_new)
            return res.rescale()

        # Case 2: Ciphertext-Ciphertext multiplication
        if isinstance(other, Ciphertext):
            self.keychain.ensure_relin_key()

            # Operations must occur at the smallest modulus of the two
            target_modulus = min(self.modulus, other.modulus)
            self_, other_ = self._equalize_moduli(other, target_modulus)

            s_c0, s_c1 = self_.c0, self_.c1
            o_c0, o_c1 = other_.c0, other_.c1

            # Step 1: Compute the tensor product 
            d0 = s_c0 * o_c0 
            d1 = s_c0 * o_c1 + s_c1 * o_c0
            d2 = s_c1 * o_c1

            # Step 2: Relinearize the d2 term.
            c0_relin, c1_relin = self._key_switch(
                d2, self.keychain.rlk, current_modulus=target_modulus
            )
            
            # Step 3: Add the relinearized components.
            c0_new = d0 + c0_relin
            c1_new = d1 + c1_relin
            
            # Step 4: Create a new ciphertext at the current level, then 
            # rescale it down.
            scale_new = self.scale_factor * other.scale_factor
            
            temp_res = self.copy(
                c0=c0_new, c1=c1_new, 
                scale_factor=scale_new, 
                modulus=target_modulus 
            )
            return temp_res.rescale()
        
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def rotate(self, rotation):
        if not isinstance(rotation, int):
            return NotImplemented

        # Ensure the rotation key is available in the context
        perm = pow(5, rotation, 2 * self.degree) if self.simd else rotation
        self.keychain.ensure_rotation_key(perm)

        # Step 1: Apply the automorphism to the ciphertext components
        c0_auto = self.c0.automorphism(perm)
        c1_auto = self.c1.automorphism(perm)

        # Step 2: Switch to correct secret key. the key switch for rotation 
        # happens at the ciphertext's current level.
        rotation_key = self.keychain.swks[perm]
        c0_ks, c1_ks = self._key_switch(c1_auto, rotation_key, 
                                        current_modulus=self.modulus)

        # Step 3: Add the component back
        c0_new = c0_auto + c0_ks 
        c1_new = c1_ks # Note: just the second part of the key switch result

        return self.copy(c0=c0_new, c1=c1_new)
        
    def _key_switch(self, poly_to_switch, switching_key, current_modulus):
        
        # 1. ModUp to the extended modulus 
        extended_modulus = current_modulus * self.keychain.modulus
        c_qq = poly_to_switch.modswitch(extended_modulus)
        
        # We need to ensure the switching key is also at this extended modulus
        swk0 = switching_key.swk0.modswitch(extended_modulus)
        swk1 = switching_key.swk1.modswitch(extended_modulus)
        
        # 2. Key Multiplication
        c0_prime = c_qq * swk0
        c1_prime = c_qq * swk1

        # 3. ModDown by the original modulus
        delta_0 = c0_prime.moddown(self.keychain.modulus)
        delta_1 = c1_prime.moddown(self.keychain.modulus)
        
        return delta_0, delta_1

    def __lshift__(self, rotation):
        return self.rotate(rotation)
    
    def __rshift__(self, rotation):
        return self.rotate(-rotation)
    
    def rescale(self, divisor=None):
        divisor = divisor or self.precision

        # Perform modular division on the polynomial coefficients and
        # reduce the scaling factor.
        c0_new = self.c0.moddown(divisor)
        c1_new = self.c1.moddown(divisor)
        new_scale = self.scale_factor // divisor
        new_modulus = c0_new.modulus

        return self.copy(
            c0=c0_new, 
            c1=c1_new, 
            scale_factor=new_scale, 
            modulus=new_modulus
        )
    
    def _equalize_moduli(self, other, target_modulus):
        new_self = self.modswitch(target_modulus)
        new_other = other.modswitch(target_modulus)
        return new_self, new_other
    
    def modswitch(self, target_modulus):
        c0_new = self.c0.modswitch(target_modulus)
        c1_new = self.c1.modswitch(target_modulus)
        
        return self.copy(
            c0=c0_new,
            c1=c1_new,
            modulus=target_modulus
        )

    def copy(self, **kwargs):
        defaults = {
            'c0': self.c0, 
            'c1': self.c1, 
            'precision': self.precision,
            'scale_factor': self.scale_factor, 
            'modulus': self.modulus,
            'keychain': self.keychain, 
            'simd': self.simd
        }
        return Ciphertext(**{**defaults, **kwargs})


class PublicKey:
    def __init__(self, pk0, pk1, modulus):
        self.pk0 = pk0
        self.pk1 = pk1
        self.modulus = modulus
    
    def __repr__(self):
        return f"PublicKey(pk0={self.pk0}, pk1={self.pk1})"


class SwitchingKey:
    def __init__(self, swk0, swk1, modulus):
        self.swk0 = swk0
        self.swk1 = swk1
        self.modulus = modulus
    
    def __repr__(self):
        return f"SwitchingKey(swk0={self.swk0}, swk1={self.swk1})"


def main():
    degree = 16
    modulus_bits = 1000
    precision_bits = 60
    
    engine = Engine(degree, modulus_bits, precision_bits)
    
    vec = np.array([1.0, 2.5, 3.0, 4.2, 5.0, 6.1, 7.8, 0.0])
    print(f"\nOriginal vector: \n{np.round(vec, 3)}")
    
    pt1 = engine.encode(vec, simd=True)
    ct1 = engine.encrypt(pt1)

    ct2 = (ct1 * ct1) + ct1
    ct3 = (ct2 * ct1) * 3.5 + (ct2 << 2) + vec[0]
    
    print("\nResulting vector:")
    print(np.round(engine.decode(engine.decrypt(ct3)).real, 3))


if __name__ == "__main__":
    main()