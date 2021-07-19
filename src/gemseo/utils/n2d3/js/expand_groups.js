function expand_collapse_group(value,svg) {
    svg.selectAll("*").remove();
    var group = parseInt(value);
    var collapsed_groups = mat.collapsed_groups;
    if (collapsed_groups.includes(group)) {
        collapsed_groups.splice(collapsed_groups.indexOf(group),1);
    }else{
        collapsed_groups.push(group);
    }
    collapsed_groups.sort();
    mat = matrix(mat.json,collapsed_groups,mat.currentOrder);
}
