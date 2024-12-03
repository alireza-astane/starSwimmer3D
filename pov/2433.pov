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
    sphere { m*<0.9418486595542592,0.564566762370262,0.4227509826132375>, 1 }        
    sphere {  m*<1.185549085334405,0.6114917799559438,3.4124661068888527>, 1 }
    sphere {  m*<3.6787962743969413,0.6114917799559436,-0.8048161016017659>, 1 }
    sphere {  m*<-2.6495217735234573,6.137937767499462,-1.7006948157680972>, 1}
    sphere { m*<-3.835270956861153,-7.753816406107958,-2.401118274895472>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.185549085334405,0.6114917799559438,3.4124661068888527>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5 }
    cylinder { m*<3.6787962743969413,0.6114917799559436,-0.8048161016017659>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5}
    cylinder { m*<-2.6495217735234573,6.137937767499462,-1.7006948157680972>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5 }
    cylinder {  m*<-3.835270956861153,-7.753816406107958,-2.401118274895472>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5}

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
    sphere { m*<0.9418486595542592,0.564566762370262,0.4227509826132375>, 1 }        
    sphere {  m*<1.185549085334405,0.6114917799559438,3.4124661068888527>, 1 }
    sphere {  m*<3.6787962743969413,0.6114917799559436,-0.8048161016017659>, 1 }
    sphere {  m*<-2.6495217735234573,6.137937767499462,-1.7006948157680972>, 1}
    sphere { m*<-3.835270956861153,-7.753816406107958,-2.401118274895472>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.185549085334405,0.6114917799559438,3.4124661068888527>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5 }
    cylinder { m*<3.6787962743969413,0.6114917799559436,-0.8048161016017659>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5}
    cylinder { m*<-2.6495217735234573,6.137937767499462,-1.7006948157680972>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5 }
    cylinder {  m*<-3.835270956861153,-7.753816406107958,-2.401118274895472>, <0.9418486595542592,0.564566762370262,0.4227509826132375>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    