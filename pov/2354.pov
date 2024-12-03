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
    sphere { m*<1.0038611270025026,0.46606741136527124,0.4594167792385412>, 1 }        
    sphere {  m*<1.2477552122819457,0.5037867368814088,3.4492467846188806>, 1 }
    sphere {  m*<3.7410024013444825,0.5037867368814086,-0.7680354238717386>, 1 }
    sphere {  m*<-2.8451632839321266,6.506016988774646,-1.8163732069010827>, 1}
    sphere { m*<-3.811257397399172,-7.822689346690189,-2.3869185974469618>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2477552122819457,0.5037867368814088,3.4492467846188806>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5 }
    cylinder { m*<3.7410024013444825,0.5037867368814086,-0.7680354238717386>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5}
    cylinder { m*<-2.8451632839321266,6.506016988774646,-1.8163732069010827>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5 }
    cylinder {  m*<-3.811257397399172,-7.822689346690189,-2.3869185974469618>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5}

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
    sphere { m*<1.0038611270025026,0.46606741136527124,0.4594167792385412>, 1 }        
    sphere {  m*<1.2477552122819457,0.5037867368814088,3.4492467846188806>, 1 }
    sphere {  m*<3.7410024013444825,0.5037867368814086,-0.7680354238717386>, 1 }
    sphere {  m*<-2.8451632839321266,6.506016988774646,-1.8163732069010827>, 1}
    sphere { m*<-3.811257397399172,-7.822689346690189,-2.3869185974469618>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2477552122819457,0.5037867368814088,3.4492467846188806>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5 }
    cylinder { m*<3.7410024013444825,0.5037867368814086,-0.7680354238717386>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5}
    cylinder { m*<-2.8451632839321266,6.506016988774646,-1.8163732069010827>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5 }
    cylinder {  m*<-3.811257397399172,-7.822689346690189,-2.3869185974469618>, <1.0038611270025026,0.46606741136527124,0.4594167792385412>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    