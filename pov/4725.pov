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
    sphere { m*<-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 1 }        
    sphere {  m*<0.41735888208581906,0.22792495524642775,6.823778530791546>, 1 }
    sphere {  m*<2.5035269039179595,-0.01678601998499396,-2.453903548058394>, 1 }
    sphere {  m*<-1.8527968499811875,2.2096539490472304,-2.198639788023181>, 1}
    sphere { m*<-1.5850096289433557,-2.678037993356667,-2.0090935028606083>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41735888208581906,0.22792495524642775,6.823778530791546>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5 }
    cylinder { m*<2.5035269039179595,-0.01678601998499396,-2.453903548058394>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5}
    cylinder { m*<-1.8527968499811875,2.2096539490472304,-2.198639788023181>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5 }
    cylinder {  m*<-1.5850096289433557,-2.678037993356667,-2.0090935028606083>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5}

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
    sphere { m*<-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 1 }        
    sphere {  m*<0.41735888208581906,0.22792495524642775,6.823778530791546>, 1 }
    sphere {  m*<2.5035269039179595,-0.01678601998499396,-2.453903548058394>, 1 }
    sphere {  m*<-1.8527968499811875,2.2096539490472304,-2.198639788023181>, 1}
    sphere { m*<-1.5850096289433557,-2.678037993356667,-2.0090935028606083>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41735888208581906,0.22792495524642775,6.823778530791546>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5 }
    cylinder { m*<2.5035269039179595,-0.01678601998499396,-2.453903548058394>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5}
    cylinder { m*<-1.8527968499811875,2.2096539490472304,-2.198639788023181>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5 }
    cylinder {  m*<-1.5850096289433557,-2.678037993356667,-2.0090935028606083>, <-0.23118149008829764,-0.1188199953713682,-1.2246940226072138>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    