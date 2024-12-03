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
    sphere { m*<-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 1 }        
    sphere {  m*<0.12744493385684724,0.12242965506229742,2.793864703881485>, 1 }
    sphere {  m*<2.621418223121417,0.09575355226834659,-1.4228995926902508>, 1 }
    sphere {  m*<-1.7349055307777363,2.3221935213005747,-1.1676358326550367>, 1}
    sphere { m*<-1.6977203693915386,-3.0014185534268076,-1.1116989256289478>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12744493385684724,0.12242965506229742,2.793864703881485>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5 }
    cylinder { m*<2.621418223121417,0.09575355226834659,-1.4228995926902508>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5}
    cylinder { m*<-1.7349055307777363,2.3221935213005747,-1.1676358326550367>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5 }
    cylinder {  m*<-1.6977203693915386,-3.0014185534268076,-1.1116989256289478>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5}

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
    sphere { m*<-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 1 }        
    sphere {  m*<0.12744493385684724,0.12242965506229742,2.793864703881485>, 1 }
    sphere {  m*<2.621418223121417,0.09575355226834659,-1.4228995926902508>, 1 }
    sphere {  m*<-1.7349055307777363,2.3221935213005747,-1.1676358326550367>, 1}
    sphere { m*<-1.6977203693915386,-3.0014185534268076,-1.1116989256289478>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12744493385684724,0.12242965506229742,2.793864703881485>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5 }
    cylinder { m*<2.621418223121417,0.09575355226834659,-1.4228995926902508>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5}
    cylinder { m*<-1.7349055307777363,2.3221935213005747,-1.1676358326550367>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5 }
    cylinder {  m*<-1.6977203693915386,-3.0014185534268076,-1.1116989256289478>, <-0.11329017088484439,-0.006280423118027789,-0.1936900672390653>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    