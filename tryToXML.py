from lxml import etree

tree = etree.parse('try.xml')
nodesA = tree.xpath('/docs/sectionA/doc')
for node in nodesA:
	print ("id = %s" % (node.text))
