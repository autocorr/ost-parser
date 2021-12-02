#!/usr/bin/env python3

import re
from pathlib import Path

import numpy as np
from lxml import (etree, objectify)


class VCI:
    p_scan = re.compile(R".+?_scan(\d+)")

    def __init__(self, fname):
        fpath = Path(fname)
        if not fpath.exists():
            raise FileNotFoundError
        self.fname = fname
        self.fpath = fpath
        self._vcitree = objectify.parse(fname)

    def __iter__(self):
        for bb in self._bb_elements:
            yield BaseBand(bb)

    def __getitem__(self, ix):
        bb = self._bb_elements[ix]
        return BaseBand(bb)

    @property
    def _bb_elements(self):
        return self._vcitree.getroot().subArray.stationInputOutput.baseBand

    @property
    def vciRequest(self):
        return self._vcitree.getroot()

    @property
    def scanId(self):
        return int(self.vciRequest.subArray.attrib["scanId"])

    @property
    def scanNum(self):
        configId = self.vciRequest.subArray.attrib["configId"]
        try:
            return int(self.p_scan.search(configId).groups()[0])
        except AttributeError:
            raise ValueError(f"Invalid scan name convention: {configId}.")

    def add(self, parent, tag):
        """Add a new element 'tag' as a subelement of parent."""
        ns = self._vcitree.getroot().nsmap["widar"]
        return objectify.SubElement(parent, f"{{ns}}{tag}")

    def getadd(self, parent, tag):
        """
        Return the requested sub-element of parent if it exists. If not, it is
        created.
        """
        try:
            return parent[tag]
        except AttributeError:
            return self.add(parent, tag)

    def write(self, fname):
        """Write the VCI file to the specified filename."""
        self._vcitree.write(
                fname,
                pretty_print=True,
                standalone=True,
                encoding="UTF-8",
        )

    def summary(self):
        out = f"{self.fname}\n"
        for bb in self:
            out += f"  {bb}\n"
            for sb in bb:
                out += f"    {sb}\n"
        return out


class BaseBand:
    def __init__(self, element):
        """
        Parameters
        ----------
        element : str
            Base band element.
        """
        self.element = element

    def __str__(self):
        return f"BB name={self.name} bb={self.bb}"

    def __iter__(self):
        for sb in self.element.subBand:
            yield SubBand(sb, self)

    def __getitem__(self, ix):
        sb = self.element.subBand[ix]
        return SubBand(sb, self)

    @property
    def swbbName(self):
        try:
            return str(self.element.attrib["swbbName"])
        except KeyError:
            return "None"

    @property
    def inQuant(self):
        return int(self.element.attrib["inQuant"])

    @property
    def is_3bit(self):
        return self.inQuant == 3

    @property
    def is_8bit(self):
        return self.inQuant == 8

    @property
    def name(self):
        try:
            return str(self.element.attrib["name"])
        except KeyError:
            return "None"

    @property
    def bw(self):
        """*unit*: Hz"""
        return float(self.element.attrib["bw"])

    @property
    def bb(self):
        return (
                int(self.element.attrib["bbA"]),
                int(self.element.attrib["bbB"]),
        )


class SubBand:
    def __init__(self, element, baseband):
        """
        Parameters
        ----------
        element : str
            Sub-band element.
        baseband : BaseBand
        """
        self.element = element
        # NOTE It would be more idiomatic to use `element.getparent()` to
        # retrieve the baseband from the subband without passing an
        # explicit reference to the `BaseBand` instance, however, these
        # relationships are not preserved when pickling because the the lxml C
        # extension for libxml does not support the pickling protocol.
        # This is primarily a problem when using `multiprocessing`.
        self.baseband = baseband

    def __str__(self):
        cbeP = "cbeP" if self.cbeproc is not None else ""
        return (
                 "SB "
                f"sw={self.swIndex:02d} "
                f"sb={self.sbid:02d} "
                f"bw={self.bw/1e6:.1f} "
                f"cf={self.cf/1e6:07.3f} "
                f"nc={self.nchan} "
                f"np={self.npol} "
                f"r={self.recirc} "
                f"t={self.minIntegTime:.4f} "
                f"cc={self.integFac['cc']} "
                f"lta={self.integFac['lta']} "
                f"cbe={self.integFac['cbe']} "
                f"ts={self.inttime:.2f} "
                f"blbs={self.blbs} "
                f"{cbeP}"
        )

    @property
    def npol(self):
        try:
            return len(self.element.polProducts.pp)
        except AttributeError:
            return 1

    @property
    def nchan(self):
        try:
            return int(self.element.polProducts.pp[0].attrib["spectralChannels"])
        except AttributeError:
            return -1

    @property
    def recirc(self):
        return int(self.element.polProducts.blbProdIntegration.attrib["recirculation"])

    @property
    def minIntegTime(self):
        """*unit*: sec"""
        min_integ = self.element.polProducts.blbProdIntegration.attrib["minIntegTime"]
        return float(min_integ) * 1e-6  # us to s

    @property
    def integFac(self):
        fac = {}
        blb_prod_integ = self.element.polProducts.blbProdIntegration.attrib
        for ff in ("cc", "lta", "cbe"):
            try:
                fac[ff] = int(blb_prod_integ[ff+"IntegFactor"])
            except KeyError:
                fac[ff] = 1
        return fac

    @property
    def inttime(self):
        """*unit*: sec"""
        fac = self.integFac
        fac_product = fac["cc"] * fac["lta"] * fac["cbe"]
        return self.minIntegTime * self.recirc * fac_product

    @property
    def bw(self):
        """*unit*: Hz"""
        return float(self.element.attrib["bw"])

    @property
    def cf(self):
        """*unit*: Hz"""
        return float(self.element.attrib["centralFreq"])

    @property
    def swIndex(self):
        return int(self.element.attrib["swIndex"])

    @property
    def sbid(self):
        return int(self.element.attrib["sbid"])

    @property
    def blbs(self):
        blblist = []
        for b in self.element.polProducts.blbPair:
            n = int(b.attrib["numBlbPairs"])
            q = int(b.attrib["quadrant"])
            p0 = int(b.attrib["firstBlbPair"])
            for i in range(n):
                blblist.append(f"q{q}p{p0+i}")
        return blblist

    @property
    def nblb(self):
        return len(self.blbs)

    @property
    def cbeproc(self):
        try:
            return self.element.polProducts.cbeProcessing.attrib
        except AttributeError:
            return None

    @property
    def freqSamp(self):
        r"""
        *unit*: Hz
        .. math:: f_{samp} = 2 f_{bw}
        """
        return 2 * self.bw

    @property
    def freqOpt(self):
        r"""
        *unit*: Hz
        .. math:: f_{opt} = \frac{1.25}{\pi} \sqrt{ \frac{f_{samp}}{\tau} }
        """
        return 1.25 / np.pi * np.sqrt(self.freqSamp / self.inttime)


def print_summary(filen):
    path = Path(filen)
    if path.exists():
        vci = VCI(filen)
        print(vci.summary())
    else:
        raise OSError(f"File not found: {filen}")


