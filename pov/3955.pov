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
    sphere { m*<-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 1 }        
    sphere {  m*<0.1125382561766084,0.0942507180801666,2.785227868902879>, 1 }
    sphere {  m*<2.6065115454411787,0.06757461528621578,-1.4315364276688562>, 1 }
    sphere {  m*<-1.7498122084579752,2.294014584318444,-1.1762726676336417>, 1}
    sphere { m*<-1.6210180096759177,-2.856423739211932,-1.0672580624437402>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1125382561766084,0.0942507180801666,2.785227868902879>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5 }
    cylinder { m*<2.6065115454411787,0.06757461528621578,-1.4315364276688562>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5}
    cylinder { m*<-1.7498122084579752,2.294014584318444,-1.1762726676336417>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5 }
    cylinder {  m*<-1.6210180096759177,-2.856423739211932,-1.0672580624437402>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5}

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
    sphere { m*<-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 1 }        
    sphere {  m*<0.1125382561766084,0.0942507180801666,2.785227868902879>, 1 }
    sphere {  m*<2.6065115454411787,0.06757461528621578,-1.4315364276688562>, 1 }
    sphere {  m*<-1.7498122084579752,2.294014584318444,-1.1762726676336417>, 1}
    sphere { m*<-1.6210180096759177,-2.856423739211932,-1.0672580624437402>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1125382561766084,0.0942507180801666,2.785227868902879>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5 }
    cylinder { m*<2.6065115454411787,0.06757461528621578,-1.4315364276688562>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5}
    cylinder { m*<-1.7498122084579752,2.294014584318444,-1.1762726676336417>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5 }
    cylinder {  m*<-1.6210180096759177,-2.856423739211932,-1.0672580624437402>, <-0.12819684856508307,-0.0344593601001586,-0.2023269022176709>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    