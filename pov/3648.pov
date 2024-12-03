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
    sphere { m*<0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 1 }        
    sphere {  m*<0.2908725609150773,0.43136615337526807,2.8885536396809246>, 1 }
    sphere {  m*<2.784845850179645,0.40469005058131713,-1.3282106568908119>, 1 }
    sphere {  m*<-1.5714779037195057,2.6311300196135443,-1.0729468968555973>, 1}
    sphere { m*<-2.412475928403958,-4.352561463140298,-1.525823786020546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2908725609150773,0.43136615337526807,2.8885536396809246>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5 }
    cylinder { m*<2.784845850179645,0.40469005058131713,-1.3282106568908119>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5}
    cylinder { m*<-1.5714779037195057,2.6311300196135443,-1.0729468968555973>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5 }
    cylinder {  m*<-2.412475928403958,-4.352561463140298,-1.525823786020546>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5}

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
    sphere { m*<0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 1 }        
    sphere {  m*<0.2908725609150773,0.43136615337526807,2.8885536396809246>, 1 }
    sphere {  m*<2.784845850179645,0.40469005058131713,-1.3282106568908119>, 1 }
    sphere {  m*<-1.5714779037195057,2.6311300196135443,-1.0729468968555973>, 1}
    sphere { m*<-2.412475928403958,-4.352561463140298,-1.525823786020546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2908725609150773,0.43136615337526807,2.8885536396809246>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5 }
    cylinder { m*<2.784845850179645,0.40469005058131713,-1.3282106568908119>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5}
    cylinder { m*<-1.5714779037195057,2.6311300196135443,-1.0729468968555973>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5 }
    cylinder {  m*<-2.412475928403958,-4.352561463140298,-1.525823786020546>, <0.050137456173385664,0.30265607519494264,-0.09900113143962608>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    