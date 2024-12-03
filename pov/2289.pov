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
    sphere { m*<1.0547425871802134,0.383404478556934,0.48950127986575687>, 1 }        
    sphere {  m*<1.2987612892499452,0.41375738071807594,3.4794052705309806>, 1 }
    sphere {  m*<3.7920084783124803,0.41375738071807583,-0.7378769379596384>, 1 }
    sphere {  m*<-3.0038103595587393,6.809177513466042,-1.9101778255905508>, 1}
    sphere { m*<-3.7906816186699754,-7.881478917337399,-2.3747517550331185>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2987612892499452,0.41375738071807594,3.4794052705309806>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5 }
    cylinder { m*<3.7920084783124803,0.41375738071807583,-0.7378769379596384>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5}
    cylinder { m*<-3.0038103595587393,6.809177513466042,-1.9101778255905508>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5 }
    cylinder {  m*<-3.7906816186699754,-7.881478917337399,-2.3747517550331185>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5}

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
    sphere { m*<1.0547425871802134,0.383404478556934,0.48950127986575687>, 1 }        
    sphere {  m*<1.2987612892499452,0.41375738071807594,3.4794052705309806>, 1 }
    sphere {  m*<3.7920084783124803,0.41375738071807583,-0.7378769379596384>, 1 }
    sphere {  m*<-3.0038103595587393,6.809177513466042,-1.9101778255905508>, 1}
    sphere { m*<-3.7906816186699754,-7.881478917337399,-2.3747517550331185>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2987612892499452,0.41375738071807594,3.4794052705309806>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5 }
    cylinder { m*<3.7920084783124803,0.41375738071807583,-0.7378769379596384>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5}
    cylinder { m*<-3.0038103595587393,6.809177513466042,-1.9101778255905508>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5 }
    cylinder {  m*<-3.7906816186699754,-7.881478917337399,-2.3747517550331185>, <1.0547425871802134,0.383404478556934,0.48950127986575687>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    