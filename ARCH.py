import math
from pathlib import Path
 
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc
 
from cross_sections import load_cross_sections

from Bfields import load_fld, ChamberWall, compute_b_inputs, plot_b_inputs

# Unit Conversions

MM = 1e-3

# System Parameters 
wallTemp = 573.15                   # K
clausing = 0.74
beamVoltage = 800                   # V
massFlow = 5.813                    # sccm
beamCurrent = 0.2084924647084211    # A  (fixed by thrust)
gridOpenArea = 0.00208492464708   # m^2
gridTotalArea = 0.0041698492941684  # m^2  (As)
transparency = 0.5   # Ts
fc = 0.1
chamberVolume = 0.000392699         # m^3
anodeWallArea = 0.012407137         # m^2  (Aas)
dischargeVoltage = 40               # V    (Vd)
keeperArea = 0.00114009182797       # m^2

# Cusp magnetics 

pointCuspField = 49 # Gauss
pointCuspField *= 1e-4
ringCuspField = 157 # Gauss
ringCuspField *= 1e-4
intercuspField = 54 # Gauss
intercuspField *= 1e-4
anodeField = 37 # Gauss
anodeField *= 1e-4

ringCuspLocation = 3.51 # cm
ringCuspLocation *= 1e-2

interCuspDiffusionLength = 1.37 # cm
interCuspDiffusionLength *= 1e-2

anodeDiffusionLength = 0.86 # cm
anodeDiffusionLength *= 1e-2
 
# Cathode / plasma potentials 
cathodeDrop = 8.0                   # V  (Vc)
plasmaPotential = 3.0               # V  (Vp, relative to anode)
 
# Energy terms
ionizationPotential = 15.76         # V  (U+, argon)        

# Fit parameters (Samples fits these to one measured operating point)
primaryReflection = 0.9             # Rp, fraction of primaries reflected at cathode cusps
# Constants
elementaryCharge = 1.60217663e-19   # C
argonMass = 6.64e-26                # kg
electronMass = 9.109e-31            # kg
avogadro = 6.022e23
stpMolarVolume = 22.413996
boltzmann = 1.380649e-23
vacuumPermittivity = 8.8541878128e-12
 
# Constant Calcs 
ionLossArea = gridTotalArea + keeperArea + fc * anodeWallArea
neutralThermalVelocity = math.sqrt(8 * boltzmann * wallTemp / (math.pi * argonMass))
massFlowNumber = massFlow * avogadro / (stpMolarVolume * 1000 * 60)
massEfficiency = beamCurrent / (elementaryCharge * massFlowNumber)
neutralDensity = (4 * beamCurrent * (1 - massEfficiency)) / (
neutralThermalVelocity * elementaryCharge * gridOpenArea * clausing * massEfficiency)
chamberPressure = neutralDensity * boltzmann * wallTemp
screenCurrent = beamCurrent * (1 - transparency) / transparency
ionTemp = boltzmann * wallTemp / elementaryCharge          # eV, for hybrid Larmor radius

# Cross sections

here = Path(__file__).resolve().parent
xsPath = here / 'Cross Sections.txt'
if not xsPath.exists():
    xsPath = here / 'Cross_Sections.txt'
db = load_cross_sections('Cross Sections.txt', ion_path='Cross Sections CEX.txt')

# Anode wall below the field-free threshold sees thermal flux instead of cross-field diffusion
anodeSegments = b.segments.label == "anode"
unshieldedFraction = b.segments.area[anodeSegments & b.segments.unshielded].sum() / b.segments.area[anodeSegments].sum()
anodeUnshieldedArea = unshieldedFraction * anodeWallArea
anodeShieldedArea = anodeWallArea - anodeUnshieldedArea

Vk = dischargeVoltage - cathodeDrop

primaryVelocity = math.sqrt(2 * elementaryCharge * Vk / electronMass)
sigmaInelasticP = db.get('inelastic', Vk)
sigmaIonP = db.get('ionization', Vk)
sigmaExcP = db.get('excitation', Vk)
screenCurrent = beamCurrent * (1 - transparency) / transparency
CEXrate = db.ion_rate(ionTemp)
Vi = neutralDensity * CEXrate





def dischargeCalc(Te, psi):
   
    plasmaIonRate = db.rate('ionization', Te)
    momentumCrossSection = db.rate('momentum', Te)
    plasmaExcRate = db.rate('excitation', Te)
    excitationEnergy = db.excitation_energy(Te)

    bohm = np.sqrt(Te * elementaryCharge / argonMass)
    electronDensity = beamCurrent / (0.6 * elementaryCharge * bohm * gridTotalArea * transparency)
    
    collisionLog = 23 - np.log(np.sqrt(electronDensity * 1e-6) / Te ** (1.5))
    Vei = 2.9e-12 * electronDensity * collisionLog / Te ** 1.5
    Ven = neutralDensity * momentumCrossSection

    Velectron = Vei + Ven
    Vion = Vi

    electronMobility = elementaryCharge / (electronMass * Velectron)
    ionMobility = elementaryCharge / (argonMass * Vion)

    crossFieldIon = ionMobility * ionTemp / (1 + ionMobility ** 2 * anodeField ** 2)
    crossFieldElectron = electronMobility * Te / (1 + electronMobility ** 2 * anodeField ** 2)
    
    ve = math.sqrt(2 * elementaryCharge * Te / electronMass)
    vi = math.sqrt(2 * elementaryCharge * ionTemp / argonMass)
    
    re1 = electronMass * ve / (elementaryCharge* pointCuspField)
    ri1 = argonMass * vi / (elementaryCharge * pointCuspField)
    hybridRad1 = 4 * np.sqrt(re1 * ri1)
    pointCuspArea = np.pi * hybridRad1 ** 2
    pointCuspCurrent = 0.6 * elementaryCharge * electronDensity * bohm * pointCuspArea

    re2 = electronMass * ve / (elementaryCharge * ringCuspField)
    ri2 = argonMass * vi / (elementaryCharge * ringCuspField)
    hybridRad2 = 4 * np.sqrt(re2 * ri2)
    ringCuspArea = ((ringCuspLocation + (hybridRad2 / 2)) ** 2 - (ringCuspLocation - (hybridRad2 / 2)) ** 2) * np.pi
    ringCuspCurrent = 0.6 * elementaryCharge * electronDensity * bohm * ringCuspArea

    b = (elementaryCharge * interCuspDiffusionLength / (electronMobility * argonMass)) * (1 + electronMobility ** 2 * intercuspField ** 2 - (Vei / Velectron))
    c = -elementaryCharge * Te / argonMass
    ambiDiffusionVelocity = -2 * c / (b + np.sqrt(b ** 2 - 4 * c))
    interCuspArea = np.pi * (ringCuspLocation + (hybridRad2 / 2)) ** 2 - pointCuspArea - ringCuspArea
    interCuspCurrent = elementaryCharge * electronDensity * ambiDiffusionVelocity * interCuspArea

    Deff = 0.1 * crossFieldElectron + 0.9 * crossFieldIon # Fitted to data in Sample's model

    anodeIonDiffusion = elementaryCharge * electronDensity * Deff * anodeShieldedArea / anodeDiffusionLength # Assumes linear density gradient
    anodeIonBohm = 0.6 * elementaryCharge * electronDensity * bohm * anodeUnshieldedArea
    anodeIonCurrent = anodeIonDiffusion + anodeIonBohm

    gammaNC = 0.2 # Assumption from Sample's model
    kMix = gammaNC * Vei / (Ven + gammaNC * Vei)
    bohmDiffusion = Te / (anodeField * 16)

    Deff2 = kMix * bohmDiffusion + (1 - kMix) * crossFieldElectron
    anodeElectronDiffusion = elementaryCharge * Deff2 * electronDensity * anodeShieldedArea / anodeDiffusionLength

    # Random thermal flux over the sheath; a negative sheath does not raise it above the random flux
    meanElectronSpeed = np.sqrt(8 * elementaryCharge * Te / (np.pi * electronMass))
    anodeElectronThermal = 0.25 * elementaryCharge * electronDensity * meanElectronSpeed * anodeUnshieldedArea * np.exp(-max(psi, 0) / Te)
    anodeElectronCurrent = anodeElectronDiffusion + anodeElectronThermal

    dischargeCurrent = anodeElectronCurrent - anodeIonCurrent - beamCurrent

    # Primary electrons (Samples Eq. 2.31 - 2.40)
    primaryCurrent = dischargeCurrent - pointCuspCurrent - ringCuspCurrent - screenCurrent

    rp1 = electronMass * primaryVelocity / (elementaryCharge * pointCuspField)
    rp2 = electronMass * primaryVelocity / (elementaryCharge * ringCuspField)
    primaryLossArea = np.pi * rp1 ** 2 + 2 * rp2 * 2 * np.pi * ringCuspLocation   # Goebel: point pi*rp^2, ring 2*rp*Lc

    anodeLossTime = chamberVolume / ((1 - primaryReflection) * primaryVelocity * primaryLossArea)
    collisionTime = 1 / (neutralDensity * primaryVelocity * (sigmaExcP + sigmaIonP))

    # Spitzer slowing of primaries on plasma electrons (NRL form, x = Vk / Te)
    coulombRate = electronDensity * elementaryCharge ** 4 * collisionLog / (
        4 * np.pi * vacuumPermittivity ** 2 * electronMass ** 2 * primaryVelocity ** 3)
    slowingTime = 1 / (2 * gammainc(1.5, Vk / Te) * coulombRate)

    primaryLifetime = 1 / (1 / collisionTime + 1 / anodeLossTime + 1 / slowingTime)
    primaryDensity = primaryCurrent * primaryLifetime / (elementaryCharge * chamberVolume)

    # Ionization and excitation (Eq. 2.21, 2.45)
    ionizationCurrent = elementaryCharge * neutralDensity * chamberVolume * (
        electronDensity * plasmaIonRate + primaryDensity * sigmaIonP * primaryVelocity)
    excitationCurrent = elementaryCharge * neutralDensity * chamberVolume * (
        electronDensity * plasmaExcRate + primaryDensity * sigmaExcP * primaryVelocity)

    # Ion conservation
    ionLossCurrent = (beamCurrent + screenCurrent + pointCuspCurrent + ringCuspCurrent
                      + interCuspCurrent + anodeIonCurrent)
    ionError = ionizationCurrent - ionLossCurrent

    # Power losses (Eq. 2.43 - 2.48)
    presheath = 0.5 * Te
    floatingSheath = Te * np.log(np.sqrt(8 * argonMass / (np.pi * electronMass)) / 2.4)   # Ie = Ii on the floating ring

    ionizationPower = ionizationPotential * ionizationCurrent
    excitationPower = excitationEnergy * excitationCurrent
    cathodeIonPower = (presheath + dischargeVoltage + psi) * (screenCurrent + pointCuspCurrent + ringCuspCurrent)
    beamIonPower = (presheath + psi) * beamCurrent                  # Eq. 2.15; Eq. 2.46 would add Vd
    anodeIonPower = (presheath + psi) * anodeIonCurrent
    anodeElectronPower = (2 * Te + psi) * anodeElectronCurrent
    interCuspPower = (presheath + floatingSheath + 2 * Te + floatingSheath) * interCuspCurrent
    primaryWallPower = elementaryCharge * primaryDensity * chamberVolume / anodeLossTime * Vk   # not in Samples 2.46 - 2.48

    powerLoss = (ionizationPower + excitationPower + cathodeIonPower + beamIonPower
                 + anodeIonPower + anodeElectronPower + interCuspPower + primaryWallPower)
    powerError = primaryCurrent * Vk - powerLoss

    return {
        'ionError': ionError,
        'powerError': powerError,
        'electronDensity': electronDensity,
        'primaryDensity': primaryDensity,
        'dischargeCurrent': dischargeCurrent,
        'primaryCurrent': primaryCurrent,
        'anodeElectronCurrent': anodeElectronCurrent,
        'anodeIonCurrent': anodeIonCurrent,
        'anodeElectronDiffusion': anodeElectronDiffusion,
        'anodeElectronThermal': anodeElectronThermal,
        'anodeIonDiffusion': anodeIonDiffusion,
        'anodeIonBohm': anodeIonBohm,
        'pointCuspCurrent': pointCuspCurrent,
        'ringCuspCurrent': ringCuspCurrent,
        'interCuspCurrent': interCuspCurrent,
        'pointCuspRadius': hybridRad1,
        'ringCuspWidth': hybridRad2,
        'ionizationPower': ionizationPower,
        'excitationPower': excitationPower,
        'cathodeIonPower': cathodeIonPower,
        'beamIonPower': beamIonPower,
        'anodeIonPower': anodeIonPower,
        'anodeElectronPower': anodeElectronPower,
        'interCuspPower': interCuspPower,
        'primaryWallPower': primaryWallPower,
    }


def sheathPotential(Te):
    # Power balance for psi at fixed Te; None if it has no root in range
    psiGrid = np.linspace(-dischargeVoltage, dischargeVoltage, 81)
    powerErrors = [dischargeCalc(Te, p)['powerError'] for p in psiGrid]
    crossings = np.where(np.diff(np.sign(powerErrors)) != 0)[0]
    if len(crossings) == 0:
        return None
    i = crossings[0]
    return brentq(lambda p: dischargeCalc(Te, p)['powerError'], psiGrid[i], psiGrid[i + 1])


def ionBalance(Te):
    psi = sheathPotential(Te)
    return np.nan if psi is None else dischargeCalc(Te, psi)['ionError']


# Solve (Samples Eq. 2.23). The thermal anode flux puts psi in the current balance,
# so psi is solved from the power balance inside the Te search on the ion balance.
TeGrid = np.linspace(0.5, 15, 60)
ionErrors = np.array([ionBalance(T) for T in TeGrid])
valid = np.isfinite(ionErrors[:-1]) & np.isfinite(ionErrors[1:])
brackets = np.where(valid & (np.diff(np.sign(ionErrors)) != 0))[0]
roots = [brentq(ionBalance, TeGrid[i], TeGrid[i + 1]) for i in brackets]
physical = [T for T in roots if dischargeCalc(T, sheathPotential(T))['primaryCurrent'] > 0]

if not physical:
    raise RuntimeError(f"No physical solution for Te in 0.5-15 eV. Ion balance roots: {np.round(roots, 2)} eV; "
                       f"largest ion error {np.nanmax(ionErrors):.3f} A at Te = {TeGrid[np.nanargmax(ionErrors)]:.2f} eV")
if len(physical) > 1:
    print(f"Multiple Te roots at {np.round(physical, 2)} eV, using the first")

Te = physical[0]
psi = sheathPotential(Te)
out = dischargeCalc(Te, psi)

dischargeLoss = dischargeVoltage * out['dischargeCurrent'] / beamCurrent
electricalEfficiency = beamCurrent * beamVoltage / (beamCurrent * beamVoltage + out['dischargeCurrent'] * dischargeVoltage)

print(f"\nResiduals: ion {out['ionError']:.2e} A, power {out['powerError']:.2e} W")
print(f"Te                  {Te:.3f} eV")
print(f"Sheath potential    {psi:.3f} V")
print(f"Plasma density      {out['electronDensity']:.3e} m^-3")
print(f"Primary density     {out['primaryDensity']:.3e} m^-3  (np/ne = {out['primaryDensity'] / out['electronDensity']:.3f})")
print(f"Neutral density     {neutralDensity:.3e} m^-3")
print(f"Discharge current   {out['dischargeCurrent']:.3f} A")
print(f"Primary current     {out['primaryCurrent']:.3f} A")
print(f"Discharge loss      {dischargeLoss:.1f} W/A")
print(f"Mass utilization    {massEfficiency:.3f}")
print(f"Electrical eff.     {electricalEfficiency:.3f}")
print(f"Unshielded anode    {100 * unshieldedFraction:.0f} % of anode area")
if psi < 0:
    print("Sheath potential is negative: model assumes a positive sheath, treat as unstable")
if out['pointCuspRadius'] > ringCuspLocation - out['ringCuspWidth'] / 2:
    print("Point and ring cusp loss areas overlap: hybrid loss width too large for these cusp fields")

print("\nCurrents [A]")
for key in ['anodeElectronCurrent', 'anodeElectronDiffusion', 'anodeElectronThermal', 'anodeIonCurrent',
            'anodeIonDiffusion', 'anodeIonBohm', 'pointCuspCurrent', 'ringCuspCurrent', 'interCuspCurrent']:
    print(f"  {key:22s} {out[key]:.4f}")

print("\nPower [W]")
for key in ['ionizationPower', 'excitationPower', 'cathodeIonPower', 'beamIonPower', 'anodeIonPower',
            'anodeElectronPower', 'interCuspPower', 'primaryWallPower']:
    print(f"  {key:22s} {out[key]:.2f}")