function matrix(json, collapsed_groups=null, initial_order="group") {

    if (collapsed_groups != null) {
        var collapsed_groups = collapsed_groups;
    }else{
        var collapsed_groups = [];
    }

    // The number of groups
    var n_groups = json.groups.length;

    var matrix = [];
    matrix.currentOrder = initial_order;

    // The nodes: [{name: name_i, group: group_i, description: desc_i}, ...]
    matrix.nodes = json.nodes;
    matrix.self_coupled_disciplines = json.self_coupled_disciplines;

    // Looking for the new positions of the original nodes.
    // Remind: the first original nodes are the groups.
    var groups_lengths = json.children.map(function(children){return children.length;});
    var new_positions = [];
    json.nodes.forEach(function(node,i){
        // For each node
        // 1. Subtract the number of groups so that the first node is a discipline
        new_positions[i] = i-n_groups;
        collapsed_groups.forEach(function(group){
            // 2.a. For each collapsed group
            if ((node.group == group) && (!node.is_group)) {
                // 2.b. If this node is a discipline belonging to the collapsed group,
                //      remove it by setting its position to -1
                new_positions[i] = -1;
            }else if (i > json.children[group][0]) {
                // 2.c. If this node is located after this collapsed group,
                //      subtract the number of disciplines belonging to this group
                //      and add 1
                new_positions[i] -= groups_lengths[group];
                new_positions[i] += 1;
            }else if ((i < json.children[group][0])&&(new_positions[i]>=0)){
                new_positions[i] += 1;
            }
        });
    });

    collapsed_groups.forEach(function(group,i){
        new_positions[group] = i;
    })

    // The links: [{source: source_i, target: target_i, value: value_i description: desc_i}, ...]
    var links = [];
    var source = null;
    var target = null;
    var i = 0;
    json.links.forEach(function(link) {
        source = link.source;
        target = link.target;
        if (collapsed_groups.includes(json.nodes[link.source].group)) {
            source = json.nodes[link.source].group;
        }
        if (collapsed_groups.includes(json.nodes[link.target].group)) {
            target = json.nodes[link.target].group;
        }
        if ((new_positions[source]>=0) && (new_positions[target]>=0)){
                links[i] = {source: new_positions[source], target: new_positions[target],
                        value: link.value, description: link.description};
            ++i;
        }
    });
    matrix.nodes = matrix.nodes.filter(
        node => ((! collapsed_groups.includes(node.group)) && (! node.is_group)) || (node.is_group && collapsed_groups.includes(node.group))
    );


    // The number of nodes
    var n = matrix.nodes.length;

    // For each node,
    // add its index as attribute,
    // and a counter initialized to 0.
    // Moreover,
    // for each node,
    // append a list of n cells (one per node) to the matrix,
    // containing the (x,y) position of the cells,
    // as well as the opacity z (initialized to 0)
    // and the description (initialized to 0).
    matrix.nodes.forEach(function(node, i) {
        node.index = i;
        node.count = 0;
        matrix[i] = d3.range(n).map(
            function(j) { return {x: j, y: i, z: 0, description: ""}; }
        );
    });

    // Fill in the matrix with the information of each link:
    // its source node, its target node, its degree and its description.
    // In addition, for each link, increase the counter of the nodes used in this link.
    links.forEach(function(link) {
      matrix[link.source][link.target].description += link.description;
      matrix[link.source][link.target].z += link.value;
      matrix.nodes[link.source].count += link.value;
      matrix.nodes[link.target].count += link.value;
    });

    var adjacency = matrix.map(function(row) {
      return row.map(function(cell) { return cell.z; });
    });

    var graph = reorder.graph()
        .nodes(matrix.nodes)
        .links(links)
        .init();

    var dist_adjacency;

    var leafOrder = reorder.optimal_leaf_order()
        .distance(reorder.distance.manhattan);

    function computeLeaforder() {
        var order = leafOrder(adjacency);
        order.forEach(function(lo, i) {
            matrix.nodes[i].leafOrder = lo;
        });
        return matrix.nodes.map(function(n) { return n.leafOrder; });
    }

    function computeLeaforderDist() {
        if (! dist_adjacency)
            dist_adjacency = reorder.graph2valuemats(graph);
        var order = reorder.valuemats_reorder(dist_adjacency, leafOrder);
        order.forEach(function(lo, i) {
            matrix.nodes[i].leafOrderDist = lo;
        });
        return matrix.nodes.map(function(n) { return n.leafOrderDist; });
    }

    function computeBarycenter() {
        var barycenter = reorder.barycenter_order(graph);
        var improved = reorder.adjacent_exchange(graph, barycenter[0], barycenter[1]);
        improved[0].forEach(function(lo, i) {
            matrix.nodes[i].barycenter = lo;
        });
        return matrix.nodes.map(function(n) { return n.barycenter; });
    }

    function computeRCM() {
        var rcm = reorder.reverse_cuthill_mckee_order(graph);
        rcm.forEach(function(lo, i) {
            matrix.nodes[i].rcm = lo;
        });
        return matrix.nodes.map(function(n) { return n.rcm; });
    }

    function computeSpectral() {
        var spectral = reorder.spectral_order(graph);
        spectral.forEach(function(lo, i) {
            matrix.nodes[i].spectral = lo;
        });
        return matrix.nodes.map(function(n) { return n.spectral; });
    }

    // Precompute the orders.
    var orders = {
        name: d3.range(n).sort(function(a, b) { return d3.ascending(matrix.nodes[a].name, matrix.nodes[b].name); }),
        count: d3.range(n).sort(function(a, b) { return matrix.nodes[b].count - matrix.nodes[a].count; }),
        group: d3.range(n).sort(function(a, b) {
            var x = matrix.nodes[b].group - matrix.nodes[a].group;
            return (x != 0) ?  x : d3.ascending(matrix.nodes[a].name, matrix.nodes[b].name);
        }),
        leafOrder: computeLeaforder,
        leafOrderDist: computeLeaforderDist,
        barycenter: computeBarycenter,
        rcm: computeRCM,
        spectral: computeSpectral
    };

  // The default sort order.
  x.domain(orders[matrix.currentOrder]);
  matrix.ordered_nodes = orders[matrix.currentOrder];

  svg.append("rect")
      .attr("class", "background")
      .attr("width", width)
      .attr("height", height);

  var row = svg.selectAll(".row")
      .data(matrix)
    .enter().append("g")
      .attr("id", function(d, i) { return "row"+i; })
      .attr("class", "row")
      .attr("transform", function(d, i) { return "translate(0," + x(i) + ")"; })
      .each(row);

  row.append("line")
      .attr("x2", width);

  row.append("text")
      .attr("id", function(d, i) { return "rowname"+i;})
      .attr("x", -6)
      .attr("y", x.rangeBand() / 2)
      .attr("dy", ".32em")
      .attr("text-anchor", "end")
      .text(function(d, i) { return matrix.nodes[i].name; })
      .attr('font-size', fontSize+'px')
      .style("cursor","help")
      .insert("title").text(function(d, i) { return matrix.nodes[i].name; });

  var column = svg.selectAll(".column")
      .data(matrix)
    .enter().append("g")
      .attr("id", function(d, i) { return "col"+i; })
      .attr("class", "column")
      .attr("transform", function(d, i) { return "translate(" + x(i) + ")rotate(-90)"; });

  column.append("line")
      .attr("x1", -width);

  column.append("text")
      .attr("id", function(d, i) { return "colname"+i;})
      .attr("x", 6)
      .attr("y", x.rangeBand() / 2)
      .attr("dy", ".32em")
      .attr("text-anchor", "start")
      .text(function(d, i) { return matrix.nodes[i].name; })
      .attr('font-size', fontSize+'px')
      .style("cursor","help")
      .insert("title").text(function(d, i) { return matrix.nodes[i].name; });

  function row(row) {
    var cell = d3.select(this).selectAll(".cell")
	  .data(row.filter(function(d) { return d.z; }))
      .enter()

    cell.append("rect")
        .attr("class", "cell")
        .attr("x", function(d) { return x(d.x); })
        .attr("width", x.rangeBand())
        .attr("height", x.rangeBand())
        .style("fill-opacity", function(d) { return (d.x == d.y) && !matrix.self_coupled_disciplines.includes(matrix.nodes[d.x].name) ? 0 : z(d.z);})
        .style("fill", function(d) { return matrix.nodes[d.x].group == matrix.nodes[d.y].group ? c(matrix.nodes[d.x].group+1) : null; })
        .on("mouseover", mouseover)
        .on("mouseout", mouseout)
        .on("click", mouseclick);

    cell.append("circle")
        .attr("class", "cell")
        .attr("cx",function(d){return x(d.x)+x.rangeBand()/2})
        .attr("cy",x.rangeBand()/2)
        .attr("r", x.rangeBand()/4)
        .style("stroke", function(d){return (d.x == d.y)? c(matrix.nodes[d.x].group+1) : null;})
        .style("stroke-width", x.rangeBand()/7.5)
        .style("fill", function(d){return c(matrix.nodes[d.x].group+1)})
        .style("fill-opacity", function(d) { return (d.x == d.y && !matrix.nodes[d.x].is_group)? 1 : 0;})
        .on("mouseover", mouseover)
        .on("mouseout", mouseout)
        .on("contextmenu",contextmenu)
        .on("click", mouseclick);
  }

  function contextmenu(d){
    d3.event.preventDefault();
    group = matrix.nodes[d.x].group
    if (group != 0){
        expand_collapse_group(group, svg);
        checked = document.getElementById("check_" + group).checked;
        document.getElementById("check_" + group).checked = !checked;
    }
  }

  function mouseclick(p) {
        var elem = document.querySelector("#matrix-sidenav");
        var instance = M.Sidenav.getInstance(elem);

        if (p.x != p.y){
            d3.select("#modal-body")
                .html(matrix[p.y][p.x].description);
        }else{
            d3.select("#modal-body")
                .html(matrix.nodes[p.x].description);
        }
        instance.open();

        window.onclick = function(event) {
          if (event.target == instance) {
            instance.close();
          }
        }
    }

    function mouseover(p) {
        var y_pos = (matrix.ordered_nodes.indexOf(p.y)+0.5)*x.rangeBand();
        var x_pos = (matrix.ordered_nodes.indexOf(p.x)+0.5)*x.rangeBand();
        d3.selectAll(".row text").classed("active", function(d, i) { return i == p.y; });
        d3.selectAll(".column text").classed("active", function(d, i) { return i == p.x; });
        d3.select(this).style("cursor", "help");
        if (p.x != p.y) {
            d3.select(this.parentElement)
            .append("rect")
            .attr("class", "highlight")
            .attr("width", width)
            .attr("height", x.rangeBand());
            d3.select("#col"+p.x)
            .append("rect")
            .attr("class", "highlight")
            .attr("x", -width)
            .attr("width", width)
            .attr("height", x.rangeBand());
            d3.select(this.parentElement.parentElement)
            .append("svg:defs").append("svg:marker")
            .attr("id", "triangle")
            .attr("refX", 6)
            .attr("refY", 6)
            .attr("markerWidth", 30)
            .attr("markerHeight", 30)
            .attr("markerUnits","userSpaceOnUse")
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M 0 0 12 6 0 12 3 6")
            .style("fill", "black");
            d3.select(this.parentElement.parentElement)
            .append('line')
            .attr("class","arrow")
            .style("stroke",  "black")
            .style("stroke-width", 3)
            .attr("x1", y_pos)
            .attr("y1", y_pos)
            .attr("x2", x_pos)
            .attr("y2", y_pos);
            d3.select(this.parentElement.parentElement)
            .append('line')
            .attr("class","arrow")
            .style("stroke", "black")
            .style("stroke-width", 3)
            .attr("x1", x_pos)
            .attr("y1", y_pos)
            .attr("x2", x_pos)
            .attr("y2", x_pos)
            .attr("marker-end", "url(#triangle)");
            d3.select(this).insert("title").text('From ' + matrix.nodes[p.y].name + '\nTo ' + matrix.nodes[p.x].name);
        }else{
           d3.select(this).insert("title").text(matrix.nodes[p.x].name);
        }
    }

    function mouseout(p) {
        d3.selectAll("text").classed("active", false);
        d3.selectAll(".arrow").remove();
        d3.selectAll(".highlight").remove();
    }

    function order(value) {
        var o = orders[value];
        matrix.currentOrder = value;
        if (typeof o === "function") {
            orders[value] = o.call();
        }
        x.domain(orders[value]);
        matrix.ordered_nodes = orders[value];

        var t = svg.transition().duration(1500);

        t.selectAll(".row")
                .delay(function(d, i) { return x(i) * 4; })
                .attr("transform", function(d, i) { return "translate(0," + x(i) + ")"; })
            .selectAll(".cell")
                .delay(function(d) { return x(d.x) * 4; })
                .attr("x", function(d) { return x(d.x); })
                .attr("cx", function(d) { return x(d.x)+x.rangeBand()/2; })
                .attr("cy", function(d) { return x.rangeBand()/2; });

        t.selectAll(".column")
                .delay(function(d, i) { return x(i) * 4; })
                .attr("transform", function(d, i) { return "translate(" + x(i) + ")rotate(-90)"; });
    }

    function distance(value) {
        leafOrder.distance(reorder.distance[value]);

        if (matrix.currentOrder == 'leafOrder') {
            orders.leafOrder = computeLeaforder;
            order("leafOrder");
        }
        else if (matrix.currentOrder == 'leafOrderDist') {
            orders.leafOrderDist = computeLeaforderDist;
            order("leafOrderDist");
    	}
    }


    matrix.order = order;
    matrix.distance = distance;

    var timeout = setTimeout(function() {}, 1000);
    matrix.timeout = timeout;
    matrix.groups = json.groups;
    matrix.collapsed_groups = collapsed_groups;
    matrix.json = json;
    return matrix;
}
