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
    sphere { m*<-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 1 }        
    sphere {  m*<0.5990377922127366,-0.36132694325121034,9.225927817527479>, 1 }
    sphere {  m*<7.954389230212709,-0.4502472192455665,-5.353565472517852>, 1 }
    sphere {  m*<-6.882637416062103,5.874172666815909,-3.732330229264588>, 1}
    sphere { m*<-2.04016082367242,-3.92765200986954,-1.2292971704798838>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5990377922127366,-0.36132694325121034,9.225927817527479>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5 }
    cylinder { m*<7.954389230212709,-0.4502472192455665,-5.353565472517852>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5}
    cylinder { m*<-6.882637416062103,5.874172666815909,-3.732330229264588>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5 }
    cylinder {  m*<-2.04016082367242,-3.92765200986954,-1.2292971704798838>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5}

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
    sphere { m*<-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 1 }        
    sphere {  m*<0.5990377922127366,-0.36132694325121034,9.225927817527479>, 1 }
    sphere {  m*<7.954389230212709,-0.4502472192455665,-5.353565472517852>, 1 }
    sphere {  m*<-6.882637416062103,5.874172666815909,-3.732330229264588>, 1}
    sphere { m*<-2.04016082367242,-3.92765200986954,-1.2292971704798838>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5990377922127366,-0.36132694325121034,9.225927817527479>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5 }
    cylinder { m*<7.954389230212709,-0.4502472192455665,-5.353565472517852>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5}
    cylinder { m*<-6.882637416062103,5.874172666815909,-3.732330229264588>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5 }
    cylinder {  m*<-2.04016082367242,-3.92765200986954,-1.2292971704798838>, <-0.824337822466199,-1.2511516864044574,-0.6322440215744984>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    