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
    sphere { m*<-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 1 }        
    sphere {  m*<0.16956120542908215,0.09543884205735281,3.7485762998465715>, 1 }
    sphere {  m*<2.5691663336349047,0.018308388616714355,-1.639309467004416>, 1 }
    sphere {  m*<-1.7871574202642424,2.2447483576489393,-1.3840457069692025>, 1}
    sphere { m*<-1.5193701992264106,-2.642943584754958,-1.1944994218066298>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16956120542908215,0.09543884205735281,3.7485762998465715>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5 }
    cylinder { m*<2.5691663336349047,0.018308388616714355,-1.639309467004416>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5}
    cylinder { m*<-1.7871574202642424,2.2447483576489393,-1.3840457069692025>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5 }
    cylinder {  m*<-1.5193701992264106,-2.642943584754958,-1.1944994218066298>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5}

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
    sphere { m*<-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 1 }        
    sphere {  m*<0.16956120542908215,0.09543884205735281,3.7485762998465715>, 1 }
    sphere {  m*<2.5691663336349047,0.018308388616714355,-1.639309467004416>, 1 }
    sphere {  m*<-1.7871574202642424,2.2447483576489393,-1.3840457069692025>, 1}
    sphere { m*<-1.5193701992264106,-2.642943584754958,-1.1944994218066298>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16956120542908215,0.09543884205735281,3.7485762998465715>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5 }
    cylinder { m*<2.5691663336349047,0.018308388616714355,-1.639309467004416>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5}
    cylinder { m*<-1.7871574202642424,2.2447483576489393,-1.3840457069692025>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5 }
    cylinder {  m*<-1.5193701992264106,-2.642943584754958,-1.1944994218066298>, <-0.16554206037135227,-0.08372558676965977,-0.41009994155323254>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    