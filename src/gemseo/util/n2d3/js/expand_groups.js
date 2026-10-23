function expand_collapse_group(value,svg) {
    svg.selectAll("*").remove();
    var group = parseInt(value);
    var collapsed_groups = mat.collapsed_groups;
    if (collapsed_groups.includes(group)) {
        collapsed_groups.splice(collapsed_groups.indexOf(group),1);
    }else{
        collapsed_groups.push(group);
        document.getElementById("check_all").checked = false;
    }
    collapsed_groups.sort();
    mat = matrix(mat.json,collapsed_groups,mat.currentOrder);
}

function expand_collapse_all(value,svg) {
    checked = document.getElementById("check_all").checked;
    for (const group of Array(value).keys()) {
        if (group != 0){
            if ((checked && !document.getElementById("check_" + group).checked) || (!checked && document.getElementById("check_" + group).checked)) {
                expand_collapse_group(group, svg);
                document.getElementById("check_" + group).checked = checked;
            }
        }
    }
}
