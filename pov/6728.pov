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
    sphere { m*<-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 1 }        
    sphere {  m*<0.38661241735023455,-0.18499858063930585,9.117677069048447>, 1 }
    sphere {  m*<7.7419638553502095,-0.2739188566336623,-5.461816220996894>, 1 }
    sphere {  m*<-5.944125478090346,4.969069234142372,-3.2532995339561093>, 1}
    sphere { m*<-2.302874300617721,-3.6353419080802003,-1.363628117100622>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.38661241735023455,-0.18499858063930585,9.117677069048447>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5 }
    cylinder { m*<7.7419638553502095,-0.2739188566336623,-5.461816220996894>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5}
    cylinder { m*<-5.944125478090346,4.969069234142372,-3.2532995339561093>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5 }
    cylinder {  m*<-2.302874300617721,-3.6353419080802003,-1.363628117100622>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5}

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
    sphere { m*<-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 1 }        
    sphere {  m*<0.38661241735023455,-0.18499858063930585,9.117677069048447>, 1 }
    sphere {  m*<7.7419638553502095,-0.2739188566336623,-5.461816220996894>, 1 }
    sphere {  m*<-5.944125478090346,4.969069234142372,-3.2532995339561093>, 1}
    sphere { m*<-2.302874300617721,-3.6353419080802003,-1.363628117100622>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.38661241735023455,-0.18499858063930585,9.117677069048447>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5 }
    cylinder { m*<7.7419638553502095,-0.2739188566336623,-5.461816220996894>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5}
    cylinder { m*<-5.944125478090346,4.969069234142372,-3.2532995339561093>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5 }
    cylinder {  m*<-2.302874300617721,-3.6353419080802003,-1.363628117100622>, <-1.0481973660042991,-0.9813462658432642,-0.7468347900491883>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    