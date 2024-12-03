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
    sphere { m*<-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 1 }        
    sphere {  m*<0.4878062652552659,0.2897656888649345,8.323142293825413>, 1 }
    sphere {  m*<2.981147454471842,-0.01828830728911629,-3.1682050573906846>, 1 }
    sphere {  m*<-2.0089851272070436,2.185839362634977,-2.593520556665158>, 1}
    sphere { m*<-1.741197906169212,-2.7018525797689206,-2.4039742715025874>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4878062652552659,0.2897656888649345,8.323142293825413>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5 }
    cylinder { m*<2.981147454471842,-0.01828830728911629,-3.1682050573906846>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5}
    cylinder { m*<-2.0089851272070436,2.185839362634977,-2.593520556665158>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5 }
    cylinder {  m*<-1.741197906169212,-2.7018525797689206,-2.4039742715025874>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5}

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
    sphere { m*<-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 1 }        
    sphere {  m*<0.4878062652552659,0.2897656888649345,8.323142293825413>, 1 }
    sphere {  m*<2.981147454471842,-0.01828830728911629,-3.1682050573906846>, 1 }
    sphere {  m*<-2.0089851272070436,2.185839362634977,-2.593520556665158>, 1}
    sphere { m*<-1.741197906169212,-2.7018525797689206,-2.4039742715025874>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4878062652552659,0.2897656888649345,8.323142293825413>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5 }
    cylinder { m*<2.981147454471842,-0.01828830728911629,-3.1682050573906846>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5}
    cylinder { m*<-2.0089851272070436,2.185839362634977,-2.593520556665158>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5 }
    cylinder {  m*<-1.741197906169212,-2.7018525797689206,-2.4039742715025874>, <-0.3814799322169181,-0.1427005131893758,-1.6296090382242105>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    