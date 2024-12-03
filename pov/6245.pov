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
    sphere { m*<-1.432909024322408,-0.450262964015855,-0.944175599529621>, 1 }        
    sphere {  m*<0.02415017718574819,0.11603467077559732,8.932967398466914>, 1 }
    sphere {  m*<7.37950161518572,0.02711439478124017,-5.646525891578442>, 1 }
    sphere {  m*<-4.159433512264207,3.142855141390628,-2.3417237777549116>, 1}
    sphere { m*<-2.7814329723667797,-3.046074927680837,-1.6086723404078869>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02415017718574819,0.11603467077559732,8.932967398466914>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5 }
    cylinder { m*<7.37950161518572,0.02711439478124017,-5.646525891578442>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5}
    cylinder { m*<-4.159433512264207,3.142855141390628,-2.3417237777549116>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5 }
    cylinder {  m*<-2.7814329723667797,-3.046074927680837,-1.6086723404078869>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5}

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
    sphere { m*<-1.432909024322408,-0.450262964015855,-0.944175599529621>, 1 }        
    sphere {  m*<0.02415017718574819,0.11603467077559732,8.932967398466914>, 1 }
    sphere {  m*<7.37950161518572,0.02711439478124017,-5.646525891578442>, 1 }
    sphere {  m*<-4.159433512264207,3.142855141390628,-2.3417237777549116>, 1}
    sphere { m*<-2.7814329723667797,-3.046074927680837,-1.6086723404078869>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02415017718574819,0.11603467077559732,8.932967398466914>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5 }
    cylinder { m*<7.37950161518572,0.02711439478124017,-5.646525891578442>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5}
    cylinder { m*<-4.159433512264207,3.142855141390628,-2.3417237777549116>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5 }
    cylinder {  m*<-2.7814329723667797,-3.046074927680837,-1.6086723404078869>, <-1.432909024322408,-0.450262964015855,-0.944175599529621>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    