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
    sphere { m*<0.2662783425513251,0.7112394327448324,0.026229534160030793>, 1 }        
    sphere {  m*<0.5070134472930168,0.8399495109251578,3.013784305280581>, 1 }
    sphere {  m*<3.0009867365575826,0.8132734081312067,-1.2029799912911527>, 1 }
    sphere {  m*<-1.3553370173415655,3.039713377163434,-0.9477162312559385>, 1}
    sphere { m*<-3.1977165887324825,-5.83694635143354,-1.9807872693149862>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5070134472930168,0.8399495109251578,3.013784305280581>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5 }
    cylinder { m*<3.0009867365575826,0.8132734081312067,-1.2029799912911527>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5}
    cylinder { m*<-1.3553370173415655,3.039713377163434,-0.9477162312559385>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5 }
    cylinder {  m*<-3.1977165887324825,-5.83694635143354,-1.9807872693149862>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5}

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
    sphere { m*<0.2662783425513251,0.7112394327448324,0.026229534160030793>, 1 }        
    sphere {  m*<0.5070134472930168,0.8399495109251578,3.013784305280581>, 1 }
    sphere {  m*<3.0009867365575826,0.8132734081312067,-1.2029799912911527>, 1 }
    sphere {  m*<-1.3553370173415655,3.039713377163434,-0.9477162312559385>, 1}
    sphere { m*<-3.1977165887324825,-5.83694635143354,-1.9807872693149862>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5070134472930168,0.8399495109251578,3.013784305280581>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5 }
    cylinder { m*<3.0009867365575826,0.8132734081312067,-1.2029799912911527>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5}
    cylinder { m*<-1.3553370173415655,3.039713377163434,-0.9477162312559385>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5 }
    cylinder {  m*<-3.1977165887324825,-5.83694635143354,-1.9807872693149862>, <0.2662783425513251,0.7112394327448324,0.026229534160030793>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    