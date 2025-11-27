import base64
import re
# Files: https://github.com/HUPO-PSI/mzML/tree/master/examples
# spec: https://www.psidev.info/mzML
import numcodecs.abc


class Base64Codec(numcodecs.abc.Codec):

    def decode(self, buf, out=None):
        buf2 = base64.b64decode(buf)
        if out:
            memoryview(out)[:] = buf2
            return out
        return buf2

    def encode(self, buf):
        return base64.b64encode(buf)


def readuntil(f, char=b"\n", blocks=1024):
    """Return data between current position and first occurrence of char

    char is included in the output, except if the end of the tile is
    encountered first.

    Parameters
    ----------
    f: filelike
    char: bytes
        Thing to find
    blocks: None or int
        How much to read in each go. Defaults to file blocksize - which may
        mean a new read on every call.
    """
    out = []
    while True:
        start = f.tell()
        part = f.read(blocks)
        if len(part) == 0:
            break
        found = part.find(char)
        if found > -1:
            out.append(part[: found + len(char)])
            f.seek(start + found + len(char))
            break
        out.append(part)
    return b"".join(out)


attr = re.compile(rb' (\S+)="(.*?)" ')
tag = re.compile(rb"<(\S+)")


def read_tag(f):
    text = readuntil(f, b">").replace(b"\n", b"").lstrip()
    start = f.tell()

    if b"<?xml" in text:
        # ignore file format line
        text = readuntil(f, b">").replace(b"\n", b"").lstrip()
    tagname = tag.search(text).groups()[0]
    if tagname[0:1] == b"/":
        # end of struct
        return text
    output = {"tagname": tagname.rstrip(b">")}
    attrs = attr.findall(text)
    for key, val in attrs:
        if val.isdigit():
            val = int(val)
        output[key] = val
    if text[-2:-1] == b"/":
        # simple tag
        return output

    # for the special case of binary, we should know how far ahead to skip from the
    # 'encodedLength' attribute of the parent tag, avoid the following block
    children = []
    while True:
        out = read_tag(f)
        if isinstance(out, bytes):
            break
        children.append(out)
    if children:
        output["children"] = children
    else:
        if tagname == b"binary>":
            output["ref"] = {"start": start, "length": len(out) - 9}
        else:
            b = out[:-(len(tagname) + 3)]
            output["value"] = int(b) if b.isdigit() else b
    return output
