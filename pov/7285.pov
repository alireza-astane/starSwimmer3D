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
    sphere { m*<-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 1 }        
    sphere {  m*<0.7426079660217603,-0.004015867272553875,9.286419797249257>, 1 }
    sphere {  m*<8.110395164344556,-0.2891081180648166,-5.284257631824678>, 1 }
    sphere {  m*<-6.7855680293444305,6.23397325555583,-3.793450728643073>, 1}
    sphere { m*<-2.7259357936940667,-5.457096130713151,-1.5119109323227213>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7426079660217603,-0.004015867272553875,9.286419797249257>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5 }
    cylinder { m*<8.110395164344556,-0.2891081180648166,-5.284257631824678>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5}
    cylinder { m*<-6.7855680293444305,6.23397325555583,-3.793450728643073>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5 }
    cylinder {  m*<-2.7259357936940667,-5.457096130713151,-1.5119109323227213>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5}

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
    sphere { m*<-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 1 }        
    sphere {  m*<0.7426079660217603,-0.004015867272553875,9.286419797249257>, 1 }
    sphere {  m*<8.110395164344556,-0.2891081180648166,-5.284257631824678>, 1 }
    sphere {  m*<-6.7855680293444305,6.23397325555583,-3.793450728643073>, 1}
    sphere { m*<-2.7259357936940667,-5.457096130713151,-1.5119109323227213>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7426079660217603,-0.004015867272553875,9.286419797249257>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5 }
    cylinder { m*<8.110395164344556,-0.2891081180648166,-5.284257631824678>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5}
    cylinder { m*<-6.7855680293444305,6.23397325555583,-3.793450728643073>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5 }
    cylinder {  m*<-2.7259357936940667,-5.457096130713151,-1.5119109323227213>, <-0.6765595281784024,-0.9939547811524718,-0.5628702997858969>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    