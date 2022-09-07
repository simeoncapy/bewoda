import nep

def createConf(node, ip="127.0.0.1", mode="one2many", port=3000):       
    return node.hybrid(ip)


def createConfDirect(node, ip="127.0.0.1", mode="one2many", port=3000):       
    return node.direct(ip, port, mode)

