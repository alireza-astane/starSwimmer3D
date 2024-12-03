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
    sphere { m*<-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 1 }        
    sphere {  m*<0.518110438692164,0.2904452950677415,8.297858029738405>, 1 }
    sphere {  m*<2.659084244743355,-0.029442483000507683,-3.0011480111351267>, 1 }
    sphere {  m*<-1.9322478831822556,2.188666204674338,-2.6289508705504017>, 1}
    sphere { m*<-1.6644606621444238,-2.6990257377295594,-2.4394045853878312>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.518110438692164,0.2904452950677415,8.297858029738405>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5 }
    cylinder { m*<2.659084244743355,-0.029442483000507683,-3.0011480111351267>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5}
    cylinder { m*<-1.9322478831822556,2.188666204674338,-2.6289508705504017>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5 }
    cylinder {  m*<-1.6644606621444238,-2.6990257377295594,-2.4394045853878312>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5}

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
    sphere { m*<-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 1 }        
    sphere {  m*<0.518110438692164,0.2904452950677415,8.297858029738405>, 1 }
    sphere {  m*<2.659084244743355,-0.029442483000507683,-3.0011480111351267>, 1 }
    sphere {  m*<-1.9322478831822556,2.188666204674338,-2.6289508705504017>, 1}
    sphere { m*<-1.6644606621444238,-2.6990257377295594,-2.4394045853878312>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.518110438692164,0.2904452950677415,8.297858029738405>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5 }
    cylinder { m*<2.659084244743355,-0.029442483000507683,-3.0011480111351267>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5}
    cylinder { m*<-1.9322478831822556,2.188666204674338,-2.6289508705504017>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5 }
    cylinder {  m*<-1.6644606621444238,-2.6990257377295594,-2.4394045853878312>, <-0.3084963852736959,-0.1398309647021568,-1.6586265790443622>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    