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
    sphere { m*<1.115567808620957,0.28247398054973666,0.5254652824957758>, 1 }        
    sphere {  m*<1.3596975981142174,0.30425188912200163,3.5154353111269314>, 1 }
    sphere {  m*<3.8529447871767535,0.3042518891220016,-0.7018468973636862>, 1 }
    sphere {  m*<-3.1916835989926304,7.1733363992862085,-2.021263467324882>, 1}
    sphere { m*<-3.7650209284632994,-7.954433210166487,-2.359578124876494>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3596975981142174,0.30425188912200163,3.5154353111269314>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5 }
    cylinder { m*<3.8529447871767535,0.3042518891220016,-0.7018468973636862>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5}
    cylinder { m*<-3.1916835989926304,7.1733363992862085,-2.021263467324882>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5 }
    cylinder {  m*<-3.7650209284632994,-7.954433210166487,-2.359578124876494>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5}

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
    sphere { m*<1.115567808620957,0.28247398054973666,0.5254652824957758>, 1 }        
    sphere {  m*<1.3596975981142174,0.30425188912200163,3.5154353111269314>, 1 }
    sphere {  m*<3.8529447871767535,0.3042518891220016,-0.7018468973636862>, 1 }
    sphere {  m*<-3.1916835989926304,7.1733363992862085,-2.021263467324882>, 1}
    sphere { m*<-3.7650209284632994,-7.954433210166487,-2.359578124876494>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3596975981142174,0.30425188912200163,3.5154353111269314>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5 }
    cylinder { m*<3.8529447871767535,0.3042518891220016,-0.7018468973636862>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5}
    cylinder { m*<-3.1916835989926304,7.1733363992862085,-2.021263467324882>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5 }
    cylinder {  m*<-3.7650209284632994,-7.954433210166487,-2.359578124876494>, <1.115567808620957,0.28247398054973666,0.5254652824957758>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    