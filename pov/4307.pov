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
    sphere { m*<-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 1 }        
    sphere {  m*<0.23325653184056938,0.1294938273115414,4.539043799478185>, 1 }
    sphere {  m*<2.5542215258307723,0.010318081665329923,-1.824776525938852>, 1 }
    sphere {  m*<-1.8021022280683747,2.2367580506975546,-1.5695127659036385>, 1}
    sphere { m*<-1.534315007030543,-2.6509338917063427,-1.3799664807410659>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23325653184056938,0.1294938273115414,4.539043799478185>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5 }
    cylinder { m*<2.5542215258307723,0.010318081665329923,-1.824776525938852>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5}
    cylinder { m*<-1.8021022280683747,2.2367580506975546,-1.5695127659036385>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5 }
    cylinder {  m*<-1.534315007030543,-2.6509338917063427,-1.3799664807410659>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5}

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
    sphere { m*<-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 1 }        
    sphere {  m*<0.23325653184056938,0.1294938273115414,4.539043799478185>, 1 }
    sphere {  m*<2.5542215258307723,0.010318081665329923,-1.824776525938852>, 1 }
    sphere {  m*<-1.8021022280683747,2.2367580506975546,-1.5695127659036385>, 1}
    sphere { m*<-1.534315007030543,-2.6509338917063427,-1.3799664807410659>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23325653184056938,0.1294938273115414,4.539043799478185>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5 }
    cylinder { m*<2.5542215258307723,0.010318081665329923,-1.824776525938852>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5}
    cylinder { m*<-1.8021022280683747,2.2367580506975546,-1.5695127659036385>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5 }
    cylinder {  m*<-1.534315007030543,-2.6509338917063427,-1.3799664807410659>, <-0.1804868681754846,-0.09171589372104424,-0.5955670004876694>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    