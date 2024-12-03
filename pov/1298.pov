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
    sphere { m*<0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 1 }        
    sphere {  m*<0.4724422799398895,-2.2105215862808097e-18,4.042330646308086>, 1 }
    sphere {  m*<7.815347831806917,3.002709679878059e-18,-1.7384318750790493>, 1 }
    sphere {  m*<-4.364787877471323,8.164965809277259,-2.1974304111947616>, 1}
    sphere { m*<-4.364787877471323,-8.164965809277259,-2.1974304111947642>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4724422799398895,-2.2105215862808097e-18,4.042330646308086>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5 }
    cylinder { m*<7.815347831806917,3.002709679878059e-18,-1.7384318750790493>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5}
    cylinder { m*<-4.364787877471323,8.164965809277259,-2.1974304111947616>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5 }
    cylinder {  m*<-4.364787877471323,-8.164965809277259,-2.1974304111947642>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5}

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
    sphere { m*<0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 1 }        
    sphere {  m*<0.4724422799398895,-2.2105215862808097e-18,4.042330646308086>, 1 }
    sphere {  m*<7.815347831806917,3.002709679878059e-18,-1.7384318750790493>, 1 }
    sphere {  m*<-4.364787877471323,8.164965809277259,-2.1974304111947616>, 1}
    sphere { m*<-4.364787877471323,-8.164965809277259,-2.1974304111947642>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4724422799398895,-2.2105215862808097e-18,4.042330646308086>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5 }
    cylinder { m*<7.815347831806917,3.002709679878059e-18,-1.7384318750790493>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5}
    cylinder { m*<-4.364787877471323,8.164965809277259,-2.1974304111947616>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5 }
    cylinder {  m*<-4.364787877471323,-8.164965809277259,-2.1974304111947642>, <0.41366432503187633,-5.6237917160206214e-18,1.042904576438599>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    