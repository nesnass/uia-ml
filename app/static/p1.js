var mydata = JSON.parse(data);
console.log(mydata[0]);

var svg = d3.select("body")
    .append("svg")
    .attr("width", 1200)
    .attr("height", 1200)
    .style("border", "1px solid black");

var imgs = svg.selectAll("image").data([0]);
for (i = 0; i < 760; i++) {
    imgs.enter()
        .append("svg:image")
        .attr("xlink:href", mydata[i].image)
        .attr("x", Math.abs(parseInt(mydata[i].x) * 32))
        .attr("y", Math.abs(parseInt(mydata[i].y) * 30))
        .attr("width", "50")
        .attr("height", "50");
}
