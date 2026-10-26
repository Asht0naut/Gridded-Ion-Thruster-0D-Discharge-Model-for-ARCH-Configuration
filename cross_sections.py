"""LXCat cross-section loader for the 0-D discharge chamber model.

Usage
-----
    from cross_sections import load_cross_sections
    db = load_cross_sections('Cross Sections.txt')                     # electrons (Biagi etc.)
    db = load_cross_sections('Cross Sections.txt', ion_path='Cross Sections CEX.txt')  # + Phelps Ar+/Ar

    db.get('ionization', 32.0)        # sigma at one energy [m^2]   (scalar or array in)
    db.rate('excitation', 4.0)        # Maxwellian <sigma v> at Te [m^3/s]
    db.excitation_energy(4.0)         # rate-weighted U* for plasma electrons [eV]
    db.excitation_energy_at(32.0)     # sigma-weighted U* for monoenergetic primaries [eV]
    db.ion_rate(0.0494)               # Ar+ - Ar momentum-transfer <sigma_mt g> [m^3/s]
    db.list()

Electron process names
----------------------
    'elastic'     elastic momentum transfer (LXCat ELASTIC block)
    'effective'   total momentum transfer, if the file has an EFFECTIVE block
    'momentum'    = 'elastic' if present, else 'effective'  (use this for nu_en)
    'ionization'  sum of all IONIZATION blocks
    'excitation'  sum of all EXCITATION blocks
    'inelastic'   ionization + excitation
    any individual excitation level by its product label, e.g. 'Ar(1S5)'

Definitions
-----------
    Maxwellian rate  <sigma v>(Te) = integral sigma(E) v(E) f(E) dE,
                     v = sqrt(2 e E / m_e),  f(E) = 2 sqrt(E/pi) Te^-1.5 exp(-E/Te)  [E, Te in eV]
    Rate-weighted excitation energy  U*(Te) = sum_j E_j <sigma_j v> / sum_j <sigma_j v>,
                     E_j = threshold of level j (energy lost per excitation)
    Ion momentum transfer  sigma_mt = sigma_iso + 2 sigma_back  (Phelps' isotropic + backscatter)
    Ion rate  <sigma_mt g>: average over the Maxwellian relative-speed distribution of ion and
              neutral (reduced mass M/2 for equal masses). LXCat's Phelps ion tables are
              indexed by centre-of-mass energy, eps = 1/2 (M/2) g^2 = E_lab/2 for equal
              masses (the file's COMMENT formulas use 2*col(A) where Phelps uses E_lab).

Table handling: linear interpolation in energy; inelastic cross sections are zero below
threshold; above the last table point the last value is held (the Maxwellian tail out
there is negligible for Te below ~30 eV).
"""
import re
from pathlib import Path

import numpy as np

E_CHARGE = 1.60217663e-19
M_E = 9.109e-31
AMU = 1.66053907e-27

_KEYWORD = re.compile(r'^[A-Z]+$')
_DASHES = re.compile(r'^-{5,}\s*$')


# --------------------------------------------------------------------------------------
# Parsing
def _parse_lxcat(path):
    """Return a list of blocks: dict(kind, target, product, param, process, species, E, sigma)."""
    lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    blocks, i, n = [], 0, len(lines)
    while i < n:
        kw = lines[i].strip()
        if _KEYWORD.match(kw) and i + 1 < n:
            target_line = lines[i + 1].strip()
            j = i + 2
            param = None
            if j < n:
                try:
                    param = float(lines[j].split()[0])
                    j += 1
                except (ValueError, IndexError):
                    pass                                # attachment blocks have no param line
        elif kw.startswith('SPECIES:'):                 # Phelps ion files: no keyword/target/param
            kw, target_line, param, j = '', '', None, i
        else:
            i += 1
            continue
        process = species = ''
        while j < n and not _DASHES.match(lines[j]):
            s = lines[j].strip()
            if s.startswith('PROCESS:'):
                process = s[8:].strip()
            elif s.startswith('SPECIES:'):
                species = s[8:].strip()
            elif _KEYWORD.match(s):                     # ran into the next block: not a real one
                break
            j += 1
        if j >= n or not _DASHES.match(lines[j]):
            i += 1
            continue
        j += 1
        rows = []
        while j < n and not _DASHES.match(lines[j]):
            parts = lines[j].split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
            j += 1
        if rows:
            E, s = np.array(rows).T
            order = np.argsort(E, kind='stable')
            if '->' in target_line or '<->' in target_line:
                tgt, prod = [t.strip() for t in re.split(r'<?->', target_line, maxsplit=1)]
            else:
                tgt, prod = target_line, ''
            blocks.append(dict(kind=kw, target=tgt, product=prod, param=param,
                               process=process, species=species, E=E[order], sigma=s[order]))
        i = j + 1
    return blocks


def _interp(E_tab, s_tab, E, threshold=None):
    E = np.asarray(E, float)
    out = np.interp(E, E_tab, s_tab, left=0.0 if threshold is not None else s_tab[0],
                    right=s_tab[-1])
    if threshold is not None:
        out = np.where(E < threshold, 0.0, out)
    return out


# --------------------------------------------------------------------------------------
class CrossSections:
    def __init__(self, path, ion_path=None, target=None, ion_mass_amu=39.948,
                 Te_grid=(0.1, 100.0, 300), n_energy=4000):
        blocks = _parse_lxcat(path)
        if target is not None:
            blocks = [b for b in blocks if b['target'] == target]
        if not blocks:
            raise ValueError(f'No LXCat cross-section blocks found in {path}')
        self.source = str(path)
        self.target = blocks[0]['target']
        self._tab = {}            # name -> (E, sigma, threshold or None)
        self.levels = {}          # excitation level label -> threshold [eV]

        exc, ion = [], []
        for b in blocks:
            k = b['kind']
            if k == 'ELASTIC':
                self._tab['elastic'] = (b['E'], b['sigma'], None)
            elif k == 'EFFECTIVE':
                self._tab['effective'] = (b['E'], b['sigma'], None)
            elif k == 'EXCITATION':
                label = b['product'] or f'exc{len(exc)}'
                self._tab[label] = (b['E'], b['sigma'], b['param'])
                self.levels[label] = b['param']
                exc.append(label)
            elif k == 'IONIZATION':
                label = 'ionization' if not ion else f'ionization{len(ion)}'
                self._tab[label + '_raw'] = (b['E'], b['sigma'], b['param'])
                ion.append(label + '_raw')
        if 'elastic' not in self._tab and 'effective' not in self._tab:
            raise ValueError('File has neither an ELASTIC nor an EFFECTIVE block.')
        if not ion:
            raise ValueError('File has no IONIZATION block.')
        self._exc_names, self._ion_names = exc, ion
        self.ionization_threshold = min(self._tab[k][2] for k in ion)

        # common energy grid for the summed processes and the Maxwellian integrals
        Emax = max(t[0][-1] for t in self._tab.values())
        thresholds = [t[2] for t in self._tab.values() if t[2] is not None]
        self._E = np.unique(np.concatenate([
            np.geomspace(1e-4, Emax, n_energy), thresholds,
            np.array(thresholds) * (1 + 1e-6)]))
        self._E = self._E[self._E > 0]
        self._sig = {}
        for name, (E, s, thr) in self._tab.items():
            self._sig[name] = _interp(E, s, self._E, thr)
        self._sig['ionization'] = sum(self._sig[k] for k in ion)
        self._sig['excitation'] = sum(self._sig[k] for k in exc) if exc else 0 * self._E
        self._sig['inelastic'] = self._sig['ionization'] + self._sig['excitation']
        # sum_j E_j sigma_j : numerator of the energy-weighted excitation loss U*
        self._sig_Uexc = sum(self.levels[k] * self._sig[k] for k in exc) if exc else 0 * self._E
        if 'elastic' in self._sig:
            self._sig['momentum'] = self._sig['elastic']
            self.momentum_source = 'elastic'
        else:
            self._sig['momentum'] = self._sig['effective']
            self.momentum_source = 'effective (includes inelastic momentum loss)'

        # Maxwellian rate tables
        self._Te = np.geomspace(*Te_grid)
        self._K = {name: self._maxwell(self._sig[name])
                   for name in ('ionization', 'excitation', 'inelastic', 'momentum',
                                'elastic', 'effective') if name in self._sig}
        self._K_Uexc = self._maxwell(self._sig_Uexc)

        self.ion = None
        if ion_path is not None:
            self._load_ions(ion_path, ion_mass_amu)

    # ---------------------------------------------------------------- electrons
    def _maxwell(self, sig):
        E = self._E
        v = np.sqrt(2 * E_CHARGE * E / M_E)
        Te = self._Te[:, None]
        f = 2 * np.sqrt(E / np.pi) * Te ** -1.5 * np.exp(-E / Te)
        return np.trapezoid(sig * v * f, E, axis=1)

    def _name(self, process):
        if process in self._sig and not process.endswith('_raw'):
            return process
        raise KeyError(f"Unknown process '{process}'. Available: {', '.join(self.names())}")

    def names(self):
        return [k for k in self._sig if not k.endswith('_raw')]

    def get(self, process, E):
        """Cross section [m^2] at electron energy E [eV] (scalar or array)."""
        name = self._name(process)
        out = np.interp(np.asarray(E, float), self._E, self._sig[name])
        return float(out) if np.ndim(E) == 0 else out

    def _rate_table(self, name):
        if name not in self._K:                         # individual level: tabulate on demand
            self._K[name] = self._maxwell(self._sig[name])
        return self._K[name]

    def rate(self, process, Te):
        """Maxwellian rate coefficient <sigma v> [m^3/s] at Te [eV] (scalar or array)."""
        return self._lookup(self._rate_table(self._name(process)), Te)

    def _lookup(self, K, Te):
        K = np.maximum(K, 1e-300)
        Te_arr = np.asarray(Te, float)
        if np.any(Te_arr < self._Te[0]) or np.any(Te_arr > self._Te[-1]):
            raise ValueError(f'Te outside tabulated range {self._Te[0]}-{self._Te[-1]} eV')
        out = np.exp(np.interp(np.log(Te_arr), np.log(self._Te), np.log(K)))
        out = np.where(out < 1e-290, 0.0, out)
        return float(out) if np.ndim(Te) == 0 else out

    def excitation_energy(self, Te):
        """Rate-weighted energy lost per excitation by Maxwellian electrons, U*(Te) [eV]."""
        num = self._lookup(self._K_Uexc, Te)
        den = self.rate('excitation', Te)
        return num / den if np.all(den > 0) else np.nan

    def excitation_energy_at(self, E):
        """Cross-section-weighted energy lost per excitation at a single energy E [eV]."""
        num = np.interp(np.asarray(E, float), self._E, self._sig_Uexc)
        den = self.get('excitation', E)
        return num / den if np.all(den > 0) else np.nan

    # ---------------------------------------------------------------- ions
    def _load_ions(self, path, mass_amu):
        """Phelps-style ion-neutral data: an isotropic and a backscatter cross section."""
        blocks = _parse_lxcat(path)
        iso = back = None
        for b in blocks:
            text = (b['kind'] + ' ' + b['process']).lower()
            if 'backscat' in text or 'back' in text:
                back = b
            elif 'isotropic' in text or 'iso' in text:
                iso = b
        if iso is None or back is None:
            found = [(b['kind'], b['process']) for b in blocks]
            raise ValueError(f'Could not identify isotropic + backscatter blocks in {path}: {found}')
        M = mass_amu * AMU
        E = np.unique(np.concatenate([iso['E'], back['E'], np.geomspace(1e-5, 1e3, 3000)]))
        E = E[E > 0]
        s_iso = _interp(iso['E'], iso['sigma'], E)
        s_back = _interp(back['E'], back['sigma'], E)
        self.ion = dict(source=str(path), M=M, E=E, iso=s_iso, back=s_back,
                        mt=s_iso + 2 * s_back)

    def ion_cross_section(self, which, E_cm):
        """Ion-neutral cross section [m^2] at centre-of-mass energy [eV] (= E_lab/2 for
        equal masses); which = 'iso','back','mt'."""
        self._need_ions()
        out = np.interp(np.asarray(E_cm, float), self.ion['E'], self.ion[which])
        return float(out) if np.ndim(E_cm) == 0 else out

    def ion_rate(self, Ti, Tn=None, which='mt'):
        """<sigma g> [m^3/s] for ions at Ti on neutrals at Tn (eV, default Tn = Ti).

        Equal masses: the relative velocity g is Maxwellian with reduced mass M/2 at
        T_rel = (Ti + Tn)/2. The table is indexed by the CM energy eps = 1/2 (M/2) g^2.
        nu_i = n0 * ion_rate(Ti).
        """
        self._need_ions()
        Tn = Ti if Tn is None else Tn
        T_rel = 0.5 * (Ti + Tn)
        eps = np.geomspace(1e-6, 60 * T_rel, 4000)                  # relative (CM) energy, eV
        f = 2 * np.sqrt(eps / np.pi) * T_rel ** -1.5 * np.exp(-eps / T_rel)
        g = np.sqrt(2 * E_CHARGE * eps / (self.ion['M'] / 2))
        sig = np.interp(eps, self.ion['E'], self.ion[which])
        return float(np.trapezoid(sig * g * f, eps))

    def _need_ions(self):
        if self.ion is None:
            raise RuntimeError('No ion-neutral data loaded: pass ion_path= to load_cross_sections.')

    # ---------------------------------------------------------------- info
    def list(self):
        print(f'Cross sections from {self.source}  (target {self.target})')
        print(f'  momentum transfer : {self.momentum_source}')
        print(f'  ionization        : {len(self._ion_names)} block(s), threshold '
              f'{self.ionization_threshold:.3f} eV')
        if self.levels:
            thr = list(self.levels.values())
            print(f'  excitation        : {len(self.levels)} levels, thresholds '
                  f'{min(thr):.3f}-{max(thr):.3f} eV')
        print(f'  Maxwellian rates  : Te {self._Te[0]:g}-{self._Te[-1]:g} eV')
        if self.ion:
            print(f'  ion-neutral       : {self.ion["source"]} (sigma_mt = iso + 2 back)')
        else:
            print('  ion-neutral       : not loaded')


def load_cross_sections(path, ion_path=None, **kw):
    return CrossSections(path, ion_path=ion_path, **kw)


# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else 'Cross Sections.txt'
    db = load_cross_sections(p, ion_path=sys.argv[2] if len(sys.argv) > 2 else None)
    db.list()

    # Gudmundsson (2002) argon Maxwellian fits, as an independent sanity check
    def K_iz(T): return 2.34e-14 * T ** 0.59 * np.exp(-17.44 / T)
    def K_ex(T): return 2.48e-14 * T ** 0.33 * np.exp(-12.78 / T)
    def K_el(T):
        L = np.log(T)
        return 2.336e-14 * T ** 1.609 * np.exp(0.0618 * L ** 2 - 0.1171 * L ** 3)

    print('\n  Te    K_iz [m3/s]  /fit   K_ex [m3/s]  /fit   K_m [m3/s]   /fit    U* [eV]')
    for T in (1.5, 2, 3, 4, 5, 7):
        print(f'{T:5.1f}   {db.rate("ionization", T):.3e} {db.rate("ionization", T) / K_iz(T):5.2f}'
              f'   {db.rate("excitation", T):.3e} {db.rate("excitation", T) / K_ex(T):5.2f}'
              f'   {db.rate("momentum", T):.3e} {db.rate("momentum", T) / K_el(T):5.2f}'
              f'   {db.excitation_energy(T):6.2f}')
    print('\n  E [eV]  sig_iz       sig_ex       sig_m        U*_at(E)')
    for E in (20, 32, 50, 100):
        print(f'{E:6d}   {db.get("ionization", E):.3e}   {db.get("excitation", E):.3e}'
              f'   {db.get("momentum", E):.3e}   {db.excitation_energy_at(E):6.2f}')