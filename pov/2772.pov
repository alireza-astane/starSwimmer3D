#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.6773168772509602,0.9555658075679557,0.2663438557719788>, 1 }        
    sphere {  m*<0.9196145179303855,1.0441281413019743,3.255227892552818>, 1 }
    sphere {  m*<3.41286170699292,1.0441281413019738,-0.9620543159377977>, 1 }
    sphere {  m*<-1.7654146727797233,4.561959083952178,-1.1779460266969546>, 1}
    sphere { m*<-3.926177664276659,-7.493932081385662,-2.4548731156380335>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9196145179303855,1.0441281413019743,3.255227892552818>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5 }
    cylinder { m*<3.41286170699292,1.0441281413019738,-0.9620543159377977>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5}
    cylinder { m*<-1.7654146727797233,4.561959083952178,-1.1779460266969546>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5 }
    cylinder {  m*<-3.926177664276659,-7.493932081385662,-2.4548731156380335>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.6773168772509602,0.9555658075679557,0.2663438557719788>, 1 }        
    sphere {  m*<0.9196145179303855,1.0441281413019743,3.255227892552818>, 1 }
    sphere {  m*<3.41286170699292,1.0441281413019738,-0.9620543159377977>, 1 }
    sphere {  m*<-1.7654146727797233,4.561959083952178,-1.1779460266969546>, 1}
    sphere { m*<-3.926177664276659,-7.493932081385662,-2.4548731156380335>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9196145179303855,1.0441281413019743,3.255227892552818>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5 }
    cylinder { m*<3.41286170699292,1.0441281413019738,-0.9620543159377977>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5}
    cylinder { m*<-1.7654146727797233,4.561959083952178,-1.1779460266969546>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5 }
    cylinder {  m*<-3.926177664276659,-7.493932081385662,-2.4548731156380335>, <0.6773168772509602,0.9555658075679557,0.2663438557719788>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    