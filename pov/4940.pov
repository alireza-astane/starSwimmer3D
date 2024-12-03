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
    sphere { m*<-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 1 }        
    sphere {  m*<0.5098238456193304,0.2773617530001625,7.971281065839101>, 1 }
    sphere {  m*<2.4752230899060557,-0.031918778118389,-2.805157656522983>, 1 }
    sphere {  m*<-1.8811006639930914,2.194521190913836,-2.5498938964877698>, 1}
    sphere { m*<-1.6133134429552596,-2.6931707514900616,-2.360347611325197>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5098238456193304,0.2773617530001625,7.971281065839101>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5 }
    cylinder { m*<2.4752230899060557,-0.031918778118389,-2.805157656522983>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5}
    cylinder { m*<-1.8811006639930914,2.194521190913836,-2.5498938964877698>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5 }
    cylinder {  m*<-1.6133134429552596,-2.6931707514900616,-2.360347611325197>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5}

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
    sphere { m*<-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 1 }        
    sphere {  m*<0.5098238456193304,0.2773617530001625,7.971281065839101>, 1 }
    sphere {  m*<2.4752230899060557,-0.031918778118389,-2.805157656522983>, 1 }
    sphere {  m*<-1.8811006639930914,2.194521190913836,-2.5498938964877698>, 1}
    sphere { m*<-1.6133134429552596,-2.6931707514900616,-2.360347611325197>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5098238456193304,0.2773617530001625,7.971281065839101>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5 }
    cylinder { m*<2.4752230899060557,-0.031918778118389,-2.805157656522983>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5}
    cylinder { m*<-1.8811006639930914,2.194521190913836,-2.5498938964877698>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5 }
    cylinder {  m*<-1.6133134429552596,-2.6931707514900616,-2.360347611325197>, <-0.2594853041002014,-0.13395275350476313,-1.5759481310718002>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    