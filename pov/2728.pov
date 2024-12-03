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
    sphere { m*<0.7110823428075969,0.9083560756699669,0.28630791892893154>, 1 }        
    sphere {  m*<0.9536142399105201,0.9914319326311204,3.2753306800259843>, 1 }
    sphere {  m*<3.4468614289730546,0.9914319326311201,-0.9419515284646303>, 1 }
    sphere {  m*<-1.885062217137552,4.766396169213855,-1.2486900322799983>, 1}
    sphere { m*<-3.9154551785648644,-7.524135676160693,-2.448532727835113>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9536142399105201,0.9914319326311204,3.2753306800259843>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5 }
    cylinder { m*<3.4468614289730546,0.9914319326311201,-0.9419515284646303>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5}
    cylinder { m*<-1.885062217137552,4.766396169213855,-1.2486900322799983>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5 }
    cylinder {  m*<-3.9154551785648644,-7.524135676160693,-2.448532727835113>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5}

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
    sphere { m*<0.7110823428075969,0.9083560756699669,0.28630791892893154>, 1 }        
    sphere {  m*<0.9536142399105201,0.9914319326311204,3.2753306800259843>, 1 }
    sphere {  m*<3.4468614289730546,0.9914319326311201,-0.9419515284646303>, 1 }
    sphere {  m*<-1.885062217137552,4.766396169213855,-1.2486900322799983>, 1}
    sphere { m*<-3.9154551785648644,-7.524135676160693,-2.448532727835113>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9536142399105201,0.9914319326311204,3.2753306800259843>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5 }
    cylinder { m*<3.4468614289730546,0.9914319326311201,-0.9419515284646303>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5}
    cylinder { m*<-1.885062217137552,4.766396169213855,-1.2486900322799983>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5 }
    cylinder {  m*<-3.9154551785648644,-7.524135676160693,-2.448532727835113>, <0.7110823428075969,0.9083560756699669,0.28630791892893154>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    